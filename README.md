# Resonant-Scanned-LiDAR
Scripts for the paper "Phase Controlled Resonant Scanner for Fast Spatial Sampling" simulation part.
arXiv link: https://arxiv.org/abs/2103.12996

In this repository, we will include three useful scripts related to the paper:
1. Resonant scanning pattern optimization with binary Regions-of-Interest (RoIs) (fast implementation).
2. Resonant scanning pattern optimization with float valued Regions-of-Interest.
3. LiDAR odometry based on resonantly scanned point cloud.

## Dependencies installation
To run the resonant scanning pattern optimization, please refer to the environment.yaml file. GPU is not required.
To run the LiDAR odometry scripts, since we adapt the scripts from the repo "LOAM" https://github.com/laboshinl/loam_velodyne. Please refer to the link for dependencies installation.

## Resonant scanning pattern optimization, binary RoI
An example is given in the jupyter notebook "Binary_RoI_Optimization.ipynb".
In this case, Regions-of-Interest (RoIs) are given by a binary map, where elements with value 1 (or other positive constants) belong to RoIs, while elements with value 0 belong to regions not of interest.

Due to the simple format of the RoI, we use an approximate nearest neighbor search (FLANN ) in the script, to increase the running speed. Usually, the optimization converges within 10-20 iterations. We implement the gradient descent optimization framework with PyTorch.


