# Kortex graspnet
This software has been developed to be used in parallel with a [Python Script](https://github.com/vislab-tecnico-lisboa/ros_kortex/blob/kinetic-devel/kortex_gazebo/scripts/grasp.py) that spawns an object to be grasped by the [Kinova Kortex Gen3](https://github.com/vislab-tecnico-lisboa/ros_kortex) 7DOF robotic arm. The software available in Kortex grapnet is directed to obtain the point cloud of the spawned object, remove some jittery points (filter the point cloud) and obtain the best grasp poses in order to successfully complete the object grasp task. Since all the available Python3 Scripts are related to the point cloud and the relative grasp pose transform, this code can easily be adapted to any other robot. 

Commands for running the scripts while operating from the [main directory](https://github.com/vislab-tecnico-lisboa/6dof-graspnet):
```shell
python -m kortex.data_extractor
python -m kortex.main
```

## data_extractor.py

Extracts, converts and filters the point cloud obtained from the camera **depth frame**. The obtained point cloud is converted and stored in an **.npz** file (data/object.npz). Always run this scripr before any other, otherwise the point cloud data will be incorrect/missing. In order to adapt this script to another robot just change the **image_depth_sub** and the **camera_info_sub** topics such that 
the depth frame captured by the camera sensor and the camera info (in order to obtain the camera intrinsics matrix) topics are both subscribed. The code is adapted for the Intel RealSense D435i camera.

## main.py

Only run this script after the object's point cloud is obtained. Generates, sortes and stores the best grasp poses, obtained using a VAE, in an **.npz** file (change the directory if implemented in other robot). Each generated grasp pose (represented by a respective transformation matrix) has an accuracy score related to it. 


## pose_viz.py

Allows the graphical visualization of both the point cloud and the generated grasps. The colour of each grasp is directly related to it's accuracy score. 

## Transfer to to other gripper

To transfer to other gripper, consider that the learning architecture was trained with the data points in the reference frame of [this gripper](https://github.com/NVlabs/6dof-graspnet/issues/8), and follow the steps described on [this issue](https://github.com/NVlabs/6dof-graspnet/issues/11#issuecomment-752876878)
