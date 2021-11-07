/*
Scripts adapted from github repos:
ALOAM: https://github.com/HKUST-Aerial-Robotics/A-LOAM
LOAM_livox: https://github.com/hku-mars/loam_livox

Major changes:
1. change ROS scripts into C++ scripts

2. In scanRegistration.cpp, make changes 
for Lissajous pattern sampled point cloud

3. In laserOdometry.cpp, make changes in feature extraction 
for Lissajous pattern sampled point cloud
*/

#pragma once

#include <cmath>

#include <pcl/point_types.h>

typedef pcl::PointXYZI PointType;

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}
