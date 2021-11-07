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

#include "config.h"
#include "kittiHelper.cpp"
#include "scanRegistration.cpp"
#include "laserOdometry.cpp"

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <boost/filesystem.hpp>

int main()
{   
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Quaterniond q_transform(R_transform);

    std::string line;
    std::size_t line_num = 0;

    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

    std::map<std::string, pcl::PointCloud<PointType>::Ptr> feature_map;

    laserOdom laser_odom;
    for (int ll = 0; ll < 800; ll++)
    {
        std::stringstream kitti_filename;
        kitti_filename << dataset_folder << std::setfill('0') << std::setw(6) << ll << ".bin";
        laserCloudIn = kitti_helper(kitti_filename.str());
        std::cout << "totally " << laserCloudIn.size() << " points in this lidar frame \n";
        std::cout << ll << std::endl;

        feature_map = scan_reg(laserCloudIn);
        laser_odom.odom_compute(feature_map);
    }
    return 0;
}
