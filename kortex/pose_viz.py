#!/usr/bin/env python
from __future__ import print_function
import argparse
import grasp_estimator
import sys
import numpy as np
import os
import tensorflow as tf
import glob
from numpy.random import randint
import mayavi.mlab as mlab
from visualization_utils import *
import mayavi.mlab as mlab
from grasp_data_reader import regularize_pc_point_count

def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y,x,:]
    
    return pc_colors

def main(args):

    # Depending on your numpy version you may need to change allow_pickle
    # from True to False.
    npy_file = "kortex/data/object.npy"
    data = np.load(npy_file, allow_pickle=True).item()
    npy_file = "kortex/data/pose_viz.npy"
    grasp_data = np.load(npy_file, allow_pickle=True).item()
    grasps = grasp_data['grasps']
    scores = grasp_data['scores']
          
    # Smoothed pc comes from averaging the depth for 10 frames and removing
    # the pixels with jittery depth between those 10 frames.
    object_pc = data['smoothed_object_pc']

    # Show Clustered Object Point Cloud
    pc = object_pc
    mlab.figure(bgcolor=(1,1,1))
    draw_scene(
        pc,
        pc_color=None,
        grasps=grasps,
        grasp_scores=scores,
        plasma_coloring=True,
        show_gripper_mesh=False,
    )
    mlab.show()

if __name__ == '__main__':
    main(sys.argv[1:])
