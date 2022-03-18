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

def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Kinova Kortex Gazebo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--vae_checkpoint_folder',
        type=str, 
        default='checkpoints/latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_')
    parser.add_argument(
        '--evaluator_checkpoint_folder', 
        type=str, 
        default='checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_/'
    )
    parser.add_argument(
        '--gradient_based_refinement',
        action='store_true',
        default=False,
    )
    parser.add_argument('--npy_folder', type=str, default='kortex/data/')
    parser.add_argument('--threshold', type=float, default=0.8)

    return parser

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


def main(args):
    parser = make_parser()
    args = parser.parse_args(args)
    cfg = grasp_estimator.joint_config(
        args.vae_checkpoint_folder,
        args.evaluator_checkpoint_folder,
    )
    cfg['threshold'] = args.threshold
    cfg['sample_based_improvement'] = 1 - int(args.gradient_based_refinement)
    cfg['num_refine_steps'] = 10 if args.gradient_based_refinement else 20
    estimator = grasp_estimator.GraspEstimator(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    sess = tf.Session()
    estimator.build_network()
    estimator.load_weights(sess)

    # Depending on your numpy version you may need to change allow_pickle
    # from True to False.
    npy_file = "kortex/data/object.npy"
    data = np.load(npy_file, allow_pickle=True).item()
    depth = data['depth']
    K = data['intrinsics_matrix']
    # Removing points that are farther than 1 meter or missing depth values.
    depth[depth == 0] = np.nan
    depth[depth > 1] = np.nan
    pc, selection = backproject(depth, K, return_finite_depth=True, return_selection=True)
          
    # Smoothed pc comes from averaging the depth for 10 frames and removing
    # the pixels with jittery depth between those 10 frames.
    object_pc = data['smoothed_object_pc']
    latents = estimator.sample_latents()
    generated_grasps, generated_scores, _ = estimator.predict_grasps(
        sess,
        object_pc,
        latents,
        num_refine_steps=cfg.num_refine_steps,
    )

    # Grasp Selection
    max_index = 0
    best_scores = np.zeros(1)
    best_grasps = np.zeros((1,4,4))
    for x in range(len(generated_scores)):
        # Find the grasp with the highest score
        if generated_scores[x] > generated_scores[max_index]:
            max_index = x
        # Find best scores
        if generated_scores[x] > 0.90:
            if best_scores.shape[0] > 1:
                best_grasps = np.concatenate((best_grasps, [generated_grasps[x]]))
                best_scores = np.concatenate((best_scores, [generated_scores[x]]))
            else:
                best_grasps[0] = generated_grasps[x]
                best_scores[0] = generated_scores[x]

    rand_index = randint(best_grasps.shape[0])
    best_grasps[0] = best_grasps[rand_index]
    best_scores[0] = best_scores[rand_index]
    while len(best_scores) > 1:
        best_array = np.delete(best_grasps, 1, 0)
        best_scores = np.delete(best_scores, 1, 0)

    if len(generated_scores) == 0:
        npy_file = "kortex/data/aaatest.npy"
        data = np.load(npy_file, allow_pickle=True).item()
        object_pc_extra = data['smoothed_object_pc']
        generated_grasps, generated_scores, _ = estimator.predict_grasps(
            sess,
            object_pc_extra,
            latents,
            num_refine_steps=cfg.num_refine_steps,
        )

    # Show Clustered Object Point Cloud
    pc = object_pc
    mlab.figure(bgcolor=(1,1,1))
    draw_scene(
        pc,
        pc_color=None,
        grasps=best_grasps,
        grasp_scores=best_scores,
        plasma_coloring=True,
        show_gripper_mesh=False,
    )
    mlab.show()

    # Save grasps and scores in file
    dict = {'grasps' : best_grasps, 'scores': best_scores}
    np.save("kortex/data/pose_viz.npy", dict)
    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("/graspnet/6dof-graspnet/kortex", "")
    path = path + "/catkin_ws_kortex/src/ros_kortex/kortex_gazebo/scripts"
    os.chdir(path)
    np.save("pose.npy", dict)
    print("\nGrasps have been saved in:" + path)

if __name__ == '__main__':
    main(sys.argv[1:])
