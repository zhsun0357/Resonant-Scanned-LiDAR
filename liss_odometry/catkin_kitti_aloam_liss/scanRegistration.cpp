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
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "config.h"
#include <map>

#include <thread>// std::this_thread::sleep_for
#include <chrono>// std::chrono::seconds

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.5;
int MAX_N_SCANS = 200;
// int MIN_SCAN_LENGTH = 20; // works for 10% density
int MIN_SCAN_LENGTH = 100; // works for original density
// int SEC_LENGTH = 100;
int MIN_SEC_LENGTH = 50;
float MIN_VIEW_ANGLE = 10 * M_PI/180;

int frame_index = 0;
double MINIMUM_RANGE = 0.1; 

float cloudCurvature[400000];
float cloudViewangle[400000];
float cloudR[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

int add_split(int idx, std::vector<int> &scanStartInd_raw, std::vector<int> &scanEndInd_raw)
{
    if(idx - scanStartInd_raw.back() <= 90)
    {
        // scan is too short, ignore
        scanStartInd_raw.pop_back();
        scanStartInd_raw.push_back(idx);
        return 0;
    }
    else
    {
        int valid_count_incre = idx - scanStartInd_raw.back();
        scanEndInd_raw.push_back(idx);
        scanStartInd_raw.push_back(idx);
        return valid_count_incre;
    }
}

std::map<std::string, pcl::PointCloud<PointType>::Ptr> scan_reg(pcl::PointCloud<pcl::PointXYZI> laserCloudIn)
{   
    TicToc t_whole;
    TicToc t_prepare;

    // start and end orientation
    int cloudSize = laserCloudIn.points.size();
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }

    bool halfPassed = false;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(MAX_N_SCANS);
    std::vector<int> scanlength; // record the length of each scan
    std::vector<int> scanStartInd_raw;
    std::vector<int> scanEndInd_raw; 

    std::vector<int> scanStartInd;
    std::vector<int> scanEndInd;
    std::vector<int> discont;

    /*
    use the criteria:
    1. turning points
    2. discontinuity
    3. if scan is too short, ignore
    */

    std::vector<float> theta, phi;
    float theta_curr, phi_curr;

    int valid_count = 0;
    int valid_count_incre;

    std::ofstream scan_f;
    std::stringstream scan_fname;
    scan_fname << PCL_DATA_SAVE_DIR << "/scan_" << frame_index << ".txt";
    scan_f.open(scan_fname.str(), std::ios::app);

    for (int ii = 0; ii < laserCloudIn.size(); ii++)
    {
        point.z = laserCloudIn.points[ii].x;
        point.x = laserCloudIn.points[ii].y;
        point.y = laserCloudIn.points[ii].z;

        theta_curr = point.y / sqrt(pow(point.x,2.0) + pow(point.z, 2.0));
        phi_curr = point.x / point.z;
        theta.push_back(theta_curr);
        phi.push_back(phi_curr);
    }

    /* split scan, with turning points and discontinuities */
    // first divide into sections with discontinuity

    discont.push_back(0);
    for(int ii = 0; ii < phi.size() - 1; ii++)
    {
        if(abs(phi[ii + 1] - phi[ii]) >= 0.01)
        {
            discont.push_back(ii + 1);
        }
    }
    discont.push_back(phi.size() - 1);
    std::sort(discont.begin(), discont.end());
    discont.erase(std::unique(discont.begin(), discont.end()), discont.end());

    for(int ss = 0; ss < discont.size() - 1; ss++)
    {
        int start = discont[ss];
        int end = discont[ss+1];

        std::vector<float> phi_sec(end - start + 1);
        // auto phi_start = phi.begin() + start;
        // auto phi_end = phi.begin() + end;
        std::copy(phi.begin() + start, phi.begin() + end, phi_sec.begin());
        // std::copy(phi_start, phi_end, phi_sec.begin());
        if(end - start > 90)
        {
            scanStartInd_raw.push_back(start);
            int num_divide = phi_sec.size() / 90;
            for(int ii = 0; ii <= num_divide; ii++)
            {
                int sec_start = std::min(90*ii, (int)phi_sec.size() - 1);
                int sec_end = std::min(90*(ii + 1), (int)phi_sec.size() - 1);
                int max_idx = std::distance(phi_sec.begin(), std::max_element(phi_sec.begin() + sec_start, phi_sec.begin() + sec_end));
                int min_idx = std::distance(phi_sec.begin(), std::min_element(phi_sec.begin() + sec_start, phi_sec.begin() + sec_end));
                int max_idx1, min_idx1;
                if((max_idx != 0) || (max_idx != (phi_sec.size() - 1)))
                {
                    int local_start = std::max(max_idx - 20, 0);
                    int local_end = std::min(max_idx + 20, (int)phi_sec.size() - 1);
                    max_idx1 = std::distance(phi_sec.begin(), std::max_element(phi_sec.begin() + local_start, phi_sec.begin() + local_end));
                }
                else
                {
                    max_idx1 = max_idx;
                }

                if((min_idx != 0) || (min_idx != (phi_sec.size() - 1)))
                {
                    int local_start = std::max(min_idx - 20, 0);
                    int local_end = std::min(min_idx + 20, (int)phi_sec.size() - 1);
                    min_idx1 = std::distance(phi_sec.begin(), std::min_element(phi_sec.begin() + local_start, phi_sec.begin() + local_end));
                }
                else
                {
                    min_idx1 = min_idx;
                }

                if((max_idx == max_idx1) && (min_idx == min_idx1))
                {
                    if(max_idx < min_idx)
                    {
                        valid_count_incre = add_split(max_idx + start, scanStartInd_raw, scanEndInd_raw);
                        valid_count += valid_count_incre;

                        valid_count_incre = add_split(min_idx + start, scanStartInd_raw, scanEndInd_raw);
                        valid_count += valid_count_incre;
                    }
                    else
                    {
                        valid_count_incre = add_split(min_idx + start, scanStartInd_raw, scanEndInd_raw);
                        valid_count += valid_count_incre;

                        valid_count_incre = add_split(max_idx + start, scanStartInd_raw, scanEndInd_raw);
                        valid_count += valid_count_incre;
                    }
                }
                else if (max_idx == max_idx1)
                {
                    valid_count_incre = add_split(max_idx + start, scanStartInd_raw, scanEndInd_raw);
                    valid_count += valid_count_incre;
                }
                else if (min_idx == min_idx1)
                {
                    valid_count_incre = add_split(min_idx + start, scanStartInd_raw, scanEndInd_raw);
                    valid_count += valid_count_incre;
                }
            }
            scanStartInd_raw.pop_back();
        }
    }

    cloudSize = valid_count;
    printf("points size %d \n", cloudSize);
    
    // sort scanStartInd_raw and scanEndInd_raw, also only take unique values
    std::sort(scanStartInd_raw.begin(), scanStartInd_raw.end());
    scanStartInd_raw.erase(std::unique(scanStartInd_raw.begin(), scanStartInd_raw.end()), scanStartInd_raw.end());
    std::sort(scanEndInd_raw.begin(), scanEndInd_raw.end());
    scanEndInd_raw.erase(std::unique(scanEndInd_raw.begin(), scanEndInd_raw.end()), scanEndInd_raw.end());

    if(scanStartInd_raw.size() != scanEndInd_raw.size())
    {
        std::cout << "scanstart, scanend does not match" << std::endl;
    }

    int num_scan = scanStartInd_raw.size();
    std::cout << "num scans " << num_scan << std::endl;

    scanStartInd_raw.resize(num_scan);
    scanEndInd_raw.resize(num_scan);

    // change all scan index to laserCloud (selected points) list
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for(int i = 0; i < num_scan; i++)
    {
        if(scanStartInd_raw[i] >= scanEndInd_raw[i])
        {
            std::cout << "scanstart, scanend does not match" << std::endl;
            // std::exit(3);
        }
        scanStartInd.push_back(laserCloud->size() + 15);
        for(int j = scanStartInd_raw[i]; j < scanEndInd_raw[i]; j++)
        {
            point = laserCloudIn.points[j];
            float relTime = (j - scanStartInd_raw[i]) / (scanEndInd_raw[i] - scanStartInd_raw[i]);
            point.intensity = i + scanPeriod * relTime;
            laserCloudScans[i].push_back(point);
        }
        *laserCloud += laserCloudScans[i];
        scanEndInd.push_back(laserCloud->size() - 16);
        scanlength.push_back(scanEndInd.back() - scanStartInd.back());
        scan_f << scanStartInd[i] << " " << scanEndInd[i] << " " << scanlength[i] << std::endl;
    }
    scan_f.close();
    laserCloudScans.resize(num_scan);
    scanlength.resize(num_scan);

    printf("prepare time %f \n", t_prepare.toc());

    std::ofstream curv_f;
    std::stringstream curv_fname;
    curv_fname << PCL_DATA_SAVE_DIR << "/curv_" << frame_index << ".txt";

    curv_f.open(curv_fname.str(), std::ios::app);
    for (int i = 1; i < cloudSize - 1; i++)
    {
        float diffX = laserCloud->points[i - 1].x + laserCloud->points[i + 1].x - 2 * laserCloud->points[i].x;
        float diffY = laserCloud->points[i - 1].y + laserCloud->points[i + 1].y - 2 * laserCloud->points[i].y;
        float diffZ = laserCloud->points[i - 1].z + laserCloud->points[i + 1].z - 2 * laserCloud->points[i].z;

        cloudCurvature[i] = diffX*diffX + diffY*diffY + diffZ*diffZ;

        Eigen::Vector3d point(laserCloud->points[i].x, laserCloud->points[i].y, laserCloud->points[i].z);
        Eigen::Vector3d point_diff(laserCloud->points[i+1].x - laserCloud->points[i-1].x, 
                                    laserCloud->points[i+1].y - laserCloud->points[i-1].y, 
                                    laserCloud->points[i+1].z - laserCloud->points[i-1].z);
        cloudViewangle[i] = acos(point.dot(point_diff) / point.norm() / point_diff.norm());

        cloudR[i] = sqrt(pow(laserCloud->points[i].x, 2.0) + pow(laserCloud->points[i].y, 2.0) + pow(laserCloud->points[i].z, 2.0));

        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
        curv_f << i << " " << cloudCurvature[i] << " " << laserCloud->points[i].x 
                << " " << laserCloud->points[i].y << " " << laserCloud->points[i].z << std::endl;
    }
    curv_f.close();

    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    for (int i = 0; i < num_scan; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 12)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);

        int num_sec = scanlength[i]/MIN_SEC_LENGTH;
        num_sec = std::min(12, num_sec);

        for (int j = 0; j < num_sec; j++)
        {   
            // sort curvature within one part of one scan line (ascending) (totally 6 frations in a scan line)
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / num_sec; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / num_sec - 1;
            int sec_length = ep - sp;

            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k];

                // other criteria apart from curvature
                bool other_check = cloudViewangle[ind] >= MIN_VIEW_ANGLE;
                float r_ori = (cloudR[ind] - cloudR[ind-1]) * (cloudR[ind] - cloudR[ind+1]);
                other_check = other_check || (r_ori > 0);
                float r_diff = std::max(abs(cloudR[ind] - cloudR[ind - 1]), abs(cloudR[ind] - cloudR[ind + 1]));
                other_check = other_check || (r_diff <= cloudR[i]*0.1);

                if ((cloudNeighborPicked[ind] == 0) &&
                    (cloudCurvature[ind] > 1.0) &&
                    (laserCloud->points[ind].x <= 40) && other_check)
                {
                    // in each fraction of each scan line, at most 2 sharpest corner feature, 20 less sharp corner feature
                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 10)
                    {                        
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; 

                    for (int l = 1; l <= 1; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.25)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -1; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.25)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if ((cloudNeighborPicked[ind] == 0) &&
                    (cloudCurvature[ind] < 0.1) &&
                    (laserCloud->points[ind].x <= 40))
                {
                    // in each scan line, at most 4 flattest surf feature, number of less flat surf feature is not limited
                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 1; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.25)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -1; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.25)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    // std::this_thread::sleep_for (std::chrono::seconds(1));
    // sleep for 1s to wait for the odometry and mapping process
    printf("scan registration time %f ms *************\n", t_whole.toc());

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfPointsFlat_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat_ptr(new pcl::PointCloud<PointType>());

    *surfPointsFlat_ptr = surfPointsFlat;
    *surfPointsLessFlat_ptr = surfPointsLessFlat;
    *cornerPointsSharp_ptr = cornerPointsSharp;
    *cornerPointsLessSharp_ptr = cornerPointsLessSharp;

    std::map<std::string, pcl::PointCloud<PointType>::Ptr> feature_map = {
        { "sharp", cornerPointsSharp_ptr},
        { "less_sharp", cornerPointsLessSharp_ptr},
        { "flat", surfPointsFlat_ptr},
        { "less_flat", surfPointsLessFlat_ptr},
        { "full", laserCloud}
    };

    std::ofstream corner_f, surf_f;
    std::stringstream corner_fname, surf_fname;
    corner_fname << PCL_DATA_SAVE_DIR << "/corner_" << frame_index << ".txt";
    surf_fname << PCL_DATA_SAVE_DIR << "/surf_" << frame_index << ".txt";

    corner_f.open(corner_fname.str(), std::ios::app);
    for(int ii = 0; ii < cornerPointsSharp.size(); ii++)
    {
        corner_f << cornerPointsSharp[ii].x << " " 
                 << cornerPointsSharp[ii].y << " "
                 << cornerPointsSharp[ii].z << std::endl;
    }
    corner_f.close();

    surf_f.open(surf_fname.str(), std::ios::app);
    for(int ii = 0; ii < surfPointsFlat.size(); ii++)
    {
        surf_f << surfPointsFlat[ii].x << " " 
               << surfPointsFlat[ii].y << " "
               << surfPointsFlat[ii].z << std::endl;
    }
    surf_f.close();

    frame_index++;
    return feature_map;
}