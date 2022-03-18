#!/usr/bin/env python
from __future__ import print_function

import roslib
import time
roslib.load_manifest('kortex_gazebo') # Change for a new robot
import sys
import os
import rospy
import cv2
import numpy as np
import message_filters as mf
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

def array_to_matrix_np(vec):

  if len(vec) == 9:
    M = np.array( [[ 616.36529541, 0, 310.25881958],[ 0, 616.20294189, 236.59980774],[ 0, 0, 1]] )
  else:
    M = np.array( [[ vec[0], vec[1], vec[2]],[ vec[3], vec[4], vec[5]],[ vec[6], vec[7], vec[8]]] )
  return M

def backproject(depth_cv, intrinsic_matrix, return_finite_depth=True, return_selection=False):

  depth = depth_cv.astype(np.float32, copy=True)

  # get intrinsic matrix
  K = intrinsic_matrix
  Kinv = np.linalg.inv(K)

  # compute the 3D points
  width = depth.shape[1]
  height = depth.shape[0]

  # construct the 2D points matrix
  x, y = np.meshgrid(np.arange(width), np.arange(height))
  ones = np.ones((height, width), dtype=np.float32)
  x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

  # backprojection
  R = np.dot(Kinv, x2d.transpose())

  # compute the 3D points
  X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
  X = np.array(X).transpose()
  if return_finite_depth:
    selection = np.isfinite(X[:, 0])
    X = X[selection, :]

  if return_selection:
    return X, selection
        
  return X

def callback(depth_sub, info):

  bridge = CvBridge()
  try:
    cv_image = bridge.imgmsg_to_cv2(depth_sub, "16UC1")
  except CvBridgeError as e:
    print(e)

  # Recizing image from 1080x720 to 640x480 (graspnet requirement)
  resized = cv2.resize(cv_image, (640, 480), interpolation= cv2.INTER_LINEAR)

  print("\nExtracting data...\n")

  # Convert CV to 'ndarray'
  depth = np.asarray(resized, dtype = np.float32)
  K = array_to_matrix_np(info.K)

  # Removing points that are farther than 1 meter or missing depth values
  depth = depth/1000
  depth[depth == 0] = np.nan
  np.where(depth > 1.0, depth, np.nan)

  smooth = depth.copy()
  smooth[smooth <= 0.18] = np.nan
  smooth[smooth >= 0.45] = np.nan
  pc = backproject(smooth, K, return_finite_depth=True, return_selection=False)

  # Delete Floor Points       
  rng = pc.shape[0]
  veclist = []
  const = 0
  for x in range(rng):
    if pc[x,1] >= 0.08 or pc[x,0] >= 0.03 or pc[x,0] <= -0.03 or pc[x,2] <= 0.16:
      veclist.append(x-const)
      const += 1
  for x in veclist:
    pc = np.delete(pc, x, 0)

  # Color is not an essencial parameter, so...
  data = np.load("kortex/data/mustard.npy", allow_pickle=True).item()
  color = data['image']

  # Save data in 'npy' file
  dict = {'depth' : depth, 'intrinsics_matrix': K, 'image' : color, 'smoothed_object_pc' : pc}
  np.save('kortex/data/object.npy', dict)

  print(depth)
  print("\nData as been successfuly extracted!\nKeyboardInterrupt to finish.")
  time.sleep(3)

def main(args):
  # Subscribe to Topics
  image_depth_sub = mf.Subscriber("realsense/depth/image_raw", Image)
  camera_info_sub = mf.Subscriber("realsense/depth/camera_info", CameraInfo)
  # Subscriptions Synchronizer
  ts = mf.TimeSynchronizer([image_depth_sub, camera_info_sub], 10)
  ts.registerCallback(callback)
  # Initialize Node
  rospy.init_node('image_converter', anonymous=True)
  # Process Execution
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  rospy.signal_shutdown("KeyboardInterrupt")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)