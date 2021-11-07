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

#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <eigen3/Eigen/Dense>
#include <queue>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include "config.h"

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

// not used
#define DISTORTION 0
constexpr double SCAN_PERIOD = 0.1;

struct OdomType{
    pcl::PointCloud<PointType>::Ptr sharp;
    pcl::PointCloud<PointType>::Ptr flat;
    pcl::PointCloud<PointType>::Ptr full;

    std::map<std::string, float> odom_map;
};

class laserOdom
{
private:
    int corner_correspondence = 0, plane_correspondence = 0;

    const double DISTANCE_SQ_THRESHOLD = 15;
    const double NEARBY_SCAN = 2.5;
    const float MIN_ANCHOR_DIS = 1e-3;

    // Transformation from current frame to world frame
    Eigen::Quaterniond q_w_curr = Eigen::Quaterniond(1,0,0,0);
    Eigen::Vector3d t_w_curr = Eigen::Vector3d(0,0,0);

    // q_curr_last(x, y, z, w), t_curr_last
    double para_q[4] = {0, 0, 0, 1};
    double para_t[3] = {0, 0, 0};

    // q_w_curr = Rk-1, t_w_curr = Tk-1
    // q_last_curr = Rk-1,k; t_last_curr = Tk-1,k
    Eigen::Map<Eigen::Quaterniond> q_last_curr = Eigen::Map<Eigen::Quaterniond>(para_q);
    // Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
    Eigen::Map<Eigen::Vector3d> t_last_curr = Eigen::Map<Eigen::Vector3d>(para_t);

    pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerLast;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfLast;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudFullRes;

    int data_save_index = 0;
    int odom_data_save_index = 0;
    bool init_flag = 0;

public: 
        
        // undistort lidar point
    void TransformToStart(PointType const *const pi, PointType *const po)
    {
        //interpolation ratio
        double s;
        if (DISTORTION)
            s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
        else
            s = 1.0;
        // slerp between identity rotation and q_last_curr
        Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
        Eigen::Vector3d t_point_last = s * t_last_curr;
        Eigen::Vector3d point(pi->x, pi->y, pi->z);
        Eigen::Vector3d un_point = q_point_last * point + t_point_last;

        po->x = un_point.x();
        po->y = un_point.y();
        po->z = un_point.z();
        po->intensity = pi->intensity;
    }

    // transform all lidar points to the start of the next frame
    void TransformToEnd(PointType const *const pi, PointType *const po)
    {
        // undistort point first
        pcl::PointXYZI un_point_tmp;
        TransformToStart(pi, &un_point_tmp);

        Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
        Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

        po->x = point_end.x();
        po->y = point_end.y();
        po->z = point_end.z();

        //Remove distortion time info
        po->intensity = int(pi->intensity);
    }

    void odom_compute(std::map<std::string, pcl::PointCloud<PointType>::Ptr> feature_map)
    {

        cornerPointsSharp = feature_map.at("sharp");
        cornerPointsLessSharp = feature_map.at("less_sharp");
        surfPointsFlat = feature_map.at("flat");
        surfPointsLessFlat = feature_map.at("less_flat");
        laserCloudFullRes = feature_map.at("full");

        // load in each point cloud features

        TicToc t_whole;
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();

        TicToc t_opt;
        if (init_flag == 0)
        {
            std::cout << "Init!" << std::endl; 
            init_flag = 1;
        }
        else
        {
            for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
            {
                corner_correspondence = 0;
                plane_correspondence = 0;

                //ceres::LossFunction *loss_function = NULL;
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization =
                    new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;

                ceres::Problem problem(problem_options);
                problem.AddParameterBlock(para_q, 4, q_parameterization);
                problem.AddParameterBlock(para_t, 3);

                pcl::PointXYZI pointSel;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                TicToc t_data;
                // find correspondence for corner features

                std::ofstream corner_anchor_f, surf_anchor_f;
                std::stringstream corner_anchor_fname, surf_anchor_fname;

                corner_anchor_fname << PCL_DATA_SAVE_DIR << "/corner_anchor_" << data_save_index << "_" << opti_counter << ".txt";
                surf_anchor_fname << PCL_DATA_SAVE_DIR << "/surf_anchor_" << data_save_index << "_" << opti_counter << ".txt";

                corner_anchor_f.open(corner_anchor_fname.str(), std::ios::app);
                surf_anchor_f.open(surf_anchor_fname.str(), std::ios::app);

                for (int i = 0; i < cornerPointsSharpNum; ++i)
                {
                    // transform each feature point to start of scan
                    // if DISTORTION is true (consider motion distortion), transform to coord at intermediate time
                    // if DISTORTION is false, just transform to coord in last frame
                    TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);

                    kdtreeCornerLast.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                    int closestPointInd = -1, minPointInd2 = -1;
                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                    {
                        closestPointInd = pointSearchInd[0];
                        int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                        // search in the direction of increasing scan line
                        for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                        {   
                            // if in the same scan line, continue
                            if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                continue;

                            // if not in nearby scans, end the loop
                            if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[j].z - pointSel.z);

                            if (pointSqDis < minPointSqDis2)
                            {
                                // find nearer point
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }

                        // search in the direction of decreasing scan line
                        for (int j = closestPointInd - 1; j >= 0; --j)
                        {
                            // if in the same scan line, continue
                            if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                continue;

                            // if not in nearby scans, end the loop
                            if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[j].z - pointSel.z);

                            if (pointSqDis < minPointSqDis2)
                            {
                                // find nearer point
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }



                    if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid, add error into cost function
                    {
                        float anchor_dis = sqrt(pow(laserCloudCornerLast->points[closestPointInd].x - laserCloudCornerLast->points[minPointInd2].x, 2.0) 
                                + pow(laserCloudCornerLast->points[closestPointInd].y - laserCloudCornerLast->points[minPointInd2].y, 2.0) 
                                + pow(laserCloudCornerLast->points[closestPointInd].z - laserCloudCornerLast->points[minPointInd2].z, 2.0));
                        if(anchor_dis > MIN_ANCHOR_DIS)
                        {
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                            corner_anchor_f << curr_point.x() << " " << curr_point.y() << " " << curr_point.z() << " "
                                            << last_point_a.x() << " " << last_point_a.y() << " " << last_point_a.z() << " "
                                            << last_point_b.x() << " " << last_point_b.y() << " " << last_point_b.z() << " "
                                            << i << " " << closestPointInd << " " << minPointInd2 << std::endl;
                        }
                    }
                }
                // find correspondence for plane features
                for (int i = 0; i < surfPointsFlatNum; ++i)
                {   
                    TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                    kdtreeSurfLast.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                    int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                    {
                        closestPointInd = pointSearchInd[0];

                        // get closest point's scan ID
                        int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                        // search in the direction of increasing scan line
                        for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                        {
                            // if not in nearby scans, end the loop
                            if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[j].z - pointSel.z);

                            // if in the same or lower scan line
                            if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            // if in the higher scan line
                            else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                            {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        // search in the direction of decreasing scan line
                        for (int j = closestPointInd - 1; j >= 0; --j)
                        {
                            // if not in nearby scans, end the loop
                            if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[j].z - pointSel.z);

                            // if in the same or higher scan line
                            if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                            {
                                // find nearer point
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        if (minPointInd2 >= 0 && minPointInd3 >= 0)
                        {

                            float anchor_dis1 = sqrt(pow(laserCloudSurfLast->points[closestPointInd].x - laserCloudSurfLast->points[minPointInd2].x, 2.0) 
                                + pow(laserCloudSurfLast->points[closestPointInd].y - laserCloudSurfLast->points[minPointInd2].y, 2.0) 
                                + pow(laserCloudSurfLast->points[closestPointInd].z - laserCloudSurfLast->points[minPointInd2].z, 2.0));

                            float anchor_dis2 = sqrt(pow(laserCloudSurfLast->points[closestPointInd].x - laserCloudSurfLast->points[minPointInd3].x, 2.0) 
                                + pow(laserCloudSurfLast->points[closestPointInd].y - laserCloudSurfLast->points[minPointInd3].y, 2.0) 
                                + pow(laserCloudSurfLast->points[closestPointInd].z - laserCloudSurfLast->points[minPointInd3].z, 2.0));
                            if((anchor_dis1 > MIN_ANCHOR_DIS) && (anchor_dis2 > MIN_ANCHOR_DIS))
                            {
                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                                surf_anchor_f << curr_point.x() << " " << curr_point.y() << " " << curr_point.z() << " "
                                                << last_point_a.x() << " " << last_point_a.y() << " " << last_point_a.z() << " "
                                                << last_point_b.x() << " " << last_point_b.y() << " " << last_point_b.z() << " "
                                                << last_point_c.x() << " " << last_point_c.y() << " " << last_point_c.z() << " "
                                                << i << " " << closestPointInd << " " << minPointInd2 << " " << minPointInd3 << std::endl;
                            }
                        }
                    }
                }
                corner_anchor_f.close();
                surf_anchor_f.close();

                printf("data association time %f ms \n", t_data.toc());

                if ((corner_correspondence + plane_correspondence) < 10)
                {
                    printf("less correspondence! *************************************************\n");
                }

                TicToc t_solver;
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                printf("solver time %f ms \n", t_solver.toc());
            }
        }
        printf("optimization twice time %f \n", t_opt.toc());

        t_w_curr = t_w_curr + q_w_curr * t_last_curr;
        q_w_curr = q_w_curr * q_last_curr;

        // record odometry (q_w_curr, t_w_curr)
        std::ofstream odom_f;
        std::stringstream odom_fname;
        odom_fname << PCL_DATA_SAVE_DIR << "/odom_" << odom_data_save_index << "_" << (odom_data_save_index + 1) << ".txt";

        // we tune the sign and order as (qx, qy, qz, qw, tx, ty, tz) and consistent coord direction with gt odom
        odom_f.open(odom_fname.str(), std::ios::app);
        odom_f << -q_w_curr.y() << " " << -q_w_curr.z() << " " << q_w_curr.x() << " " << q_w_curr.w() << " " << -t_w_curr.y() << " " \
                << -t_w_curr.z() << " " << t_w_curr.x() << std::endl;

        odom_f.close();
        odom_data_save_index++;
        data_save_index++;

        TicToc t_pub;
        // transform corner features and plane features to the scan end point

        // swap the previous feature points with the current feature points
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        // cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;
        laserCloudTemp = surfPointsLessFlat;
        // surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        kdtreeCornerLast.setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast.setInputCloud(laserCloudSurfLast);

        printf("publication time %f ms \n", t_pub.toc());
        printf("whole laserOdometry time %f ms \n \n", t_whole.toc());

    }
};

