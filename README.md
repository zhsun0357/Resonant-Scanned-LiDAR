# Resonant-Scanned-LiDAR
Scripts for the paper "Phase Controlled Resonant Scanner for Fast Spatial Sampling" simulation part.
arXiv link: https://arxiv.org/abs/2103.12996

In this repository, we will include three useful scripts related to the paper:
1. Resonant scanning pattern optimization with binary Regions-of-Interest (RoIs) (fast implementation).
2. Resonant scanning pattern optimization with float valued Regions-of-Interest.
3. LiDAR odometry based on resonantly scanned point cloud (coming soon).

## Dependencies installation
To run the resonant scanning pattern optimization, please refer to the environment.yaml file. GPU is not required.

To run the LiDAR odometry scripts, since we adapt the scripts from the repo "LOAM" https://github.com/laboshinl/loam_velodyne. Please refer to the link for dependencies installation.

## Resonant scanning pattern optimization, binary RoI
An example is given in the jupyter notebook "Binary_RoI_Optimization.ipynb".

In this case, Regions-of-Interest (RoIs) are given by a binary map, where elements with value 1 (or other positive constants) belong to RoIs, while elements with value 0 belong to regions not of interest.

Due to the simple format of the RoI, we use an approximate nearest neighbor search (FLANN https://github.com/flann-lib/flann) in the script, to increase the running speed. Usually, the optimization converges within 10-20 iterations. We implement the gradient descent optimization framework with PyTorch.

## Resonant scanning pattern optimization, float RoI
An example is given in the jupyter notebook "Float_RoI_Optimization.py".

Different from the binary RoI case, here the RoIs are given through a float value "weight map". Each element in the weight map indicates the importance of corresponding spatial location. A spatial location is assigned a higher value if it is more important.
Fast implementation is not provided in this case and the optimizatin converges have complicated dependencies on the initialization condition and the weight map definition.

## LiDAR odometry with resonantly scanned point cloud
Please follow the jupyter notebook "liss_odometry/Lissajous_odometry_simulations.ipynb".
Before running the scripts, please first download the necessary dataset files into the folder "liss_odometry":
+ KITTI point cloud (raster scanned): https://drive.google.com/drive/folders/1Nqq_Pj9WePrrU_BXc9hrPtmklTQMyvZm?usp=sharing
+ KITTI dense depth map (estimated using depth completion algorithm: https://github.com/fangchangma/self-supervised-depth-completion): https://drive.google.com/drive/folders/1rUaSU5rmpaXjFsNPkUi57S5XymRQ16N8?usp=sharing


