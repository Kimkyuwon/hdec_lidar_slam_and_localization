#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <queue>
#include <fstream>
#include <csignal>
#include <optional>
#include <unistd.h>
#include <condition_variable>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/console/print.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/Vector3.h>
#include <common_lib.h>
#include "solid/solid_module.h"
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
///// Nano-GICP
#include <nano_gicp/point_type_nano_gicp.hpp>
#include <nano_gicp/nano_gicp.hpp>

using namespace std;

using namespace gtsam;

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

struct Pose6 {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

string save_directory, optimized_poseDirectory, odom_poseDirectory, ScansDirectory, DebugDirectory;
int curr_kf_idx = 0;
vector<nav_msgs::Odometry> kf_poses;
SOLiDModule solidModule;
std::mutex mLoop, mViz, mBuf, mKF;
condition_variable sig_buffer;

// solid params
double R_SOLiD_THRES;
double FOV_u, FOV_d, VOXEL_SIZE;
int NUM_ANGLE, NUM_RANGE, NUM_HEIGHT;
int MIN_DISTANCE, MAX_DISTANCE, NUM_EXCLUDE_RECENT, NUM_CANDIDATES_FROM_TREE;
queue<tuple<int, int, Eigen::Matrix4f>> solidLoopBuf; 

// edge measurement params
pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter_map;
nano_gicp::NanoGICP<PointType2, PointType2> gicp;
std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;
pcl::KdTreeFLANN<PointType>::Ptr kdtree (new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree2 (new pcl::KdTreeFLANN<pcl::PointXYZI>());
std::vector<int> indiceLet;
double dop_thres = 0;

//for pose graph
gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
bool isLoopClosed = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
gtsam::Vector odomNoiseVector6(6);
gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
int recentIdxUpdated = 0;

//range image 
std::vector<std::vector<double>> scan_range_data;
std::vector<std::vector<double>> map_range_data;
int horizontal_resolution = static_cast<int>(2*M_PI/0.02);
double LIDAR_HOR_MIN = -180.0F;
double LIDAR_HOR_MAX = 180.0F;

visualization_msgs::Marker loopLine;
nav_msgs::Path PGO_path;
pcl::PointCloud<pcl::PointXYZI>::Ptr MapCloud(new pcl::PointCloud<pcl::PointXYZI>());

fstream odom_stream, optimized_stream;
pcl::PointCloud<pcl::PointXYZI> kf_nodes;
std::vector<Pose6> keyframePoses;
std::vector<Pose6> keyframePosesUpdated;
std::vector<double> keyframeTimes;
ros::Publisher kf_node_pub;
ros::Publisher LoopLineMarker_pub;
ros::Publisher PubPGO_path;
ros::Publisher PubPGO_map;
image_transport::Publisher PubScan_range; 
image_transport::Publisher PubMap_range;

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1.0), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );
} // initNoises

gtsam::Pose3 Pose6toGTSAMPose3(const Pose6& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6toGTSAMPose3

Pose6 getOdom(nav_msgs::Odometry _odom)
{
    auto tx = _odom.pose.pose.position.x;
    auto ty = _odom.pose.pose.position.y;
    auto tz = _odom.pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom.pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6{tx, ty, tz, roll, pitch, yaw};
} // getOdom
void insertPoint(const pcl::PointXYZI& pt, std::vector<std::vector<double>> &ri) 
{
    double range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    if (range < 0.2) return;

    double azimuth = std::atan2(pt.y, pt.x);
    double elevation = std::asin(pt.z / range);

    double vertical_fov_min = FOV_d*M_PI/180; 
    double vertical_fov_max = FOV_u*M_PI/180;
    double horizontal_fov_min = LIDAR_HOR_MIN*M_PI/180;    
    double horizontal_fov_max = LIDAR_HOR_MAX*M_PI/180;
    if (azimuth < horizontal_fov_min || azimuth > horizontal_fov_max ||
        elevation < vertical_fov_min || elevation > vertical_fov_max) return;

    int col = static_cast<int>((azimuth - horizontal_fov_min) / (horizontal_fov_max - horizontal_fov_min) * horizontal_resolution);
    int row = static_cast<int>((elevation - vertical_fov_min) / (vertical_fov_max - vertical_fov_min) * NUM_HEIGHT);

    if (row < 0) row = 0;
    if (row >= NUM_HEIGHT) row = NUM_HEIGHT - 1;
    if (col < 0) col = 0;
    if (col >= horizontal_resolution) col = horizontal_resolution - 1;

    if (range < ri[row][col]) 
    {
        ri[row][col] = range;
    }
}

cv::Mat toColorizedCVImage(std::vector<std::vector<double>> &ri) 
{
    cv::Mat img(NUM_HEIGHT, horizontal_resolution, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < NUM_HEIGHT; ++i) {
        for (int j = 0; j < horizontal_resolution; ++j) {
            if (!std::isinf(ri[i][j])) {
                double norm_range = std::min(ri[i][j] / static_cast<double>(MAX_DISTANCE), 1.0);
                img.at<uchar>(i, j) = static_cast<uchar>((1.0 - norm_range) * 255);
            }
        }
    }

    cv::Mat color_img;
    cv::applyColorMap(img, color_img, cv::COLORMAP_JET);
    cv::flip(color_img, color_img, 0); // 0은 상하 반전
    return color_img;
}

Eigen::Matrix4f get_TF_Matrix(const Pose6 Pose)
{
    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(Pose.yaw, Eigen::Vector3f::UnitZ())
             * Eigen::AngleAxisf(Pose.pitch, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(Pose.roll, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f TF(Eigen::Matrix4f::Identity());
    TF.block(0,0,3,3) = rotation;
    TF(0,3) = Pose.x;
    TF(1,3) = Pose.y;
    TF(2,3) = Pose.z;

    return TF;
}

void kf_callback(const Frame::ConstPtr &kf_msg) 
{
    mBuf.lock();
    Frame curr_kf = *kf_msg;
    curr_kf_idx = curr_kf.frame_idx;
    nav_msgs::Odometry curr_pose = curr_kf.pose;
    kf_poses.push_back(curr_pose);

    Pose6 pose_curr = getOdom(curr_pose);
    keyframePoses.push_back(pose_curr);
    keyframePosesUpdated.push_back(pose_curr);

    sensor_msgs::PointCloud2 pc_msg = curr_kf.pointcloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr curr_kf_pc (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr curr_kf_pc_down (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(pc_msg, *curr_kf_pc);
    solidModule.down_sampling(*curr_kf_pc, curr_kf_pc_down);
    solidModule.makeAndSaveSolid(*curr_kf_pc_down);
            
    // scan_range_data.clear();
    // scan_range_data.resize(NUM_HEIGHT, std::vector<double>(horizontal_resolution, std::numeric_limits<double>::infinity()));
    // for (const auto& pt : curr_kf_pc_down->points)
    // {
    //     insertPoint(pt, scan_range_data);
    // }
    // cv::Mat img = toColorizedCVImage(scan_range_data);
    // sensor_msgs::ImagePtr scanRange_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    // scanRange_msg->header.stamp = curr_pose.header.stamp;
    // scanRange_msg->header.frame_id = "camera_init";    
    // PubScan_range.publish(scanRange_msg);

    // map_range_data.clear();
    // map_range_data.resize(NUM_HEIGHT, std::vector<double>(horizontal_resolution, std::numeric_limits<double>::infinity()));
    
    // Eigen::Matrix4f curr_TF = get_TF_Matrix(pose_curr);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr curr_MapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    // pcl::transformPointCloud(*MapCloud, *curr_MapCloud, curr_TF.inverse());
    // for (const auto& pt : curr_MapCloud->points)
    // {
    //     insertPoint(pt, map_range_data);
    // }
    // cv::Mat map_img = toColorizedCVImage(map_range_data);
    // sensor_msgs::ImagePtr mapRange_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", map_img).toImageMsg();
    // mapRange_msg->header.stamp = curr_pose.header.stamp;
    // mapRange_msg->header.frame_id = "camera_init";    
    // PubMap_range.publish(mapRange_msg);

         
    pcl::io::savePCDFileBinary(ScansDirectory + to_string(curr_kf_idx) + ".pcd", *curr_kf_pc_down); // scan data
    

    const int prev_node_idx = keyframePoses.size() - 2;
    const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
    
    if(!gtSAMgraphMade)
    {
        const int init_node_idx = 0;
        gtsam::Pose3 poseOrigin = Pose6toGTSAMPose3(keyframePoses.at(init_node_idx));

        // prior factor
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
        initialEstimate.insert(init_node_idx, poseOrigin);

        gtSAMgraphMade = true;
    }
    else
    {
        gtsam::Pose3 poseFrom = Pose6toGTSAMPose3(keyframePoses.at(prev_node_idx));
        gtsam::Pose3 poseTo = Pose6toGTSAMPose3(keyframePoses.at(curr_node_idx));
        // odom factor
        gtsam::Pose3 relPose = poseFrom.between(poseTo);

        odomNoiseVector6 << kf_poses[curr_node_idx].pose.covariance[0], 
                            kf_poses[curr_node_idx].pose.covariance[7], 
                            kf_poses[curr_node_idx].pose.covariance[14], 
                            kf_poses[curr_node_idx].pose.covariance[21], 
                            kf_poses[curr_node_idx].pose.covariance[28], 
                            kf_poses[curr_node_idx].pose.covariance[35];
        odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);
        

        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relPose, odomNoise));
        initialEstimate.insert(curr_node_idx, poseTo);
    } 

    odom_stream << curr_kf.header.stamp.toSec() << " "
                << curr_pose.pose.pose.position.x << " " << curr_pose.pose.pose.position.y << " " << curr_pose.pose.pose.position.z << " " 
                << curr_pose.pose.pose.orientation.x << " " << curr_pose.pose.pose.orientation.y << " " << curr_pose.pose.pose.orientation.z << " " << curr_pose.pose.pose.orientation.w <<endl;
    
    pcl::PointXYZI kf_node;
    kf_node.x = curr_pose.pose.pose.position.x;
    kf_node.y = curr_pose.pose.pose.position.y;
    kf_node.z = curr_pose.pose.pose.position.z;
    kf_node.intensity = curr_kf_idx;
    kf_nodes.push_back(kf_node);

    sensor_msgs::PointCloud2 kf_nodes_msg;
    pcl::toROSMsg(kf_nodes, kf_nodes_msg);
    kf_nodes_msg.header.stamp = curr_pose.header.stamp;
    kf_nodes_msg.header.frame_id = "camera_init";
    kf_node_pub.publish(kf_nodes_msg);  

    double timeLaserOdometry = curr_kf.pose.header.stamp.toSec();
    keyframeTimes.push_back(timeLaserOdometry);
    mBuf.unlock();
}

void updatePoses(void)
{
    PGO_path.poses.clear();
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
        keyframePosesUpdated[node_idx] = p;

        geometry_msgs::PoseStamped poseStampPGO;
        poseStampPGO.header.frame_id = "camera_init";
        poseStampPGO.pose.position.x = p.x;
        poseStampPGO.pose.position.y = p.y;
        poseStampPGO.pose.position.z = p.z;
        poseStampPGO.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(p.roll, p.pitch, p.yaw);
        PGO_path.header.frame_id = "camera_init";
        PGO_path.poses.push_back(poseStampPGO);
        PGO_path.poses[node_idx].header.stamp = poseStampPGO.header.stamp;
    }
    PubPGO_path.publish(PGO_path);
}

void runISAM2opt(void)
{
    // called when a variable added
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (isLoopClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isLoopClosed = false;
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    recentIdxUpdated = int(isamCurrentEstimate.size());
    updatePoses();
}

void performSOLiDLoopClosure(void)
{
    if( int(keyframePoses.size()) < solidModule.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = solidModule.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SOLiDclosestHistoryFrameID = std::get<1>(detectResult);
    if( SOLiDclosestHistoryFrameID != -1 ) 
    { 
        const int prev_node_idx = SOLiDclosestHistoryFrameID;
        const int curr_node_idx = std::get<0>(detectResult); // because cpp starts 0 and ends n-1
        Eigen::Vector3d dist_vec;
        dist_vec(0) = keyframePoses[curr_node_idx].x - keyframePoses[prev_node_idx].x;
        dist_vec(1) = keyframePoses[curr_node_idx].y - keyframePoses[prev_node_idx].y;
        dist_vec(2) = keyframePoses[curr_node_idx].z - keyframePoses[prev_node_idx].z;
        double dist = dist_vec.norm();

        if (dist > 50.0) return;

        // Eigen::Matrix4f to_TF = get_TF_Matrix(keyframePoses[curr_node_idx]);
        // Eigen::Matrix4f from_TF = get_TF_Matrix(keyframePoses[prev_node_idx]);
        // Eigen::Matrix4f delta_TF = from_TF.inverse() * to_TF;
        Eigen::Matrix4f delta_TF (Eigen::Matrix4f::Identity());
        Eigen::Matrix3f rotation;
            rotation = Eigen::AngleAxisf(std::get<2>(detectResult), Eigen::Vector3f::UnitZ())
                     * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())
                     * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());
        delta_TF.block(0,0,3,3) = rotation;

        if (solidLoopBuf.size() <= 10)
        {
            solidLoopBuf.push(std::make_tuple(prev_node_idx, curr_node_idx, delta_TF));
        }        
    }
} // performSOLiDLoopClosure

std::optional<gtsam::Pose3> doGICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, Eigen::Matrix4f delta_TF)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr targetKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_curr_kf_idx) + ".pcd", *cureKeyframeCloud);
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_loop_kf_idx) + ".pcd", *targetKeyframeCloud);

    gicp.setInputTarget(targetKeyframeCloud);
    // gicp.calculateSourceCovariances();
    gicp.setInputSource(cureKeyframeCloud);
    // gicp.calculateTargetCovariances();

    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    gicp.align(*aligned_cloud, delta_TF);
    Eigen::Matrix4f edge_TF = gicp.getFinalTransformation();
    pcl::PointCloud<pcl::PointXYZI>::Ptr matchKeyframeCloud (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*cureKeyframeCloud, *matchKeyframeCloud, edge_TF);

    Eigen::Matrix<double, 6, 6> hessian = gicp.getHessian();
    
    typedef Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> EigenSolver;
    EigenSolver es;
    Eigen::Matrix<double, 6, 6> hessian_inv = hessian.inverse();
    es.compute(hessian_inv);
    
    Eigen::VectorXcd eigenvalues = es.eigenvalues();

    std::complex<double> max_eigenvalue = eigenvalues(0);
    for (int i = 1; i < eigenvalues.size(); ++i) 
    {
        if (eigenvalues(i).real() > max_eigenvalue.real()) {

            max_eigenvalue = eigenvalues(i);
        }
    }
    double max_eigen = sqrt(fabs(max_eigenvalue.real()));

    kdtree2->setInputCloud(targetKeyframeCloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr MatchedCloud (new pcl::PointCloud<pcl::PointXYZI>());
    for (int k = 0; k < matchKeyframeCloud->points.size(); k++)
    {
        kdtree2->nearestKSearch(matchKeyframeCloud->points[k], 1, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < 0.1)
        {
            MatchedCloud->points.push_back(matchKeyframeCloud->points[k]);
        }
    }

    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setInputCloud(MatchedCloud);
    downSizeFilter.setLeafSize(2.0, 2.0, 2.0);
    downSizeFilter.setMinimumPointsNumberPerVoxel(2);
    downSizeFilter.filter(*MatchedCloud);
    pcl::removeNaNFromPointCloud(*MatchedCloud, *MatchedCloud, indiceLet);
    indiceLet.clear();
    std::vector<Eigen::Vector3d> range_info;
    for (size_t k = 0; k < MatchedCloud->points.size(); k++)
    {
        double r = sqrt(pow((MatchedCloud->points[k].x), 2) + pow((MatchedCloud->points[k].y), 2) + pow((MatchedCloud->points[k].z), 2));
        if (r < 1.0)
        {
            continue;
        }
        Eigen::Vector3d r_info;
        r_info(0) = (MatchedCloud->points[k].x) / r;
        r_info(1) = (MatchedCloud->points[k].y) / r;
        r_info(2) = (MatchedCloud->points[k].z) / r;
        range_info.push_back(r_info);
    }
    Eigen::MatrixXd AA(range_info.size(), 3);
    for (size_t p = 0; p < range_info.size(); p++)
    {
        AA(p, 0) = range_info[p](0);
        AA(p, 1) = range_info[p](1);
        AA(p, 2) = range_info[p](2);
    }
    Eigen::Matrix3d A_sq;
    Eigen::Matrix3d Q;
    A_sq = AA.transpose() * AA;
    Q = A_sq.inverse();
    double pdop = sqrt(Q(0, 0) + Q(1, 1) + Q(2, 2));
    if (pdop == 0 || pdop > 100 || std::isnan(pdop) == true)
    {
        pdop = 100;
    }    

    pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, delta_TF);
    std::for_each(cureKeyframeCloud->points.begin(), cureKeyframeCloud->points.end(),
                  [](pcl::PointXYZI& point) { point.intensity = 1.0; });
    std::for_each(targetKeyframeCloud->points.begin(), targetKeyframeCloud->points.end(),
                  [](pcl::PointXYZI& point) { point.intensity = 2.0; });
    std::for_each(matchKeyframeCloud->points.begin(), matchKeyframeCloud->points.end(),
                  [](pcl::PointXYZI& point) { point.intensity = 3.0; });

    pcl::PointCloud<pcl::PointXYZI>::Ptr resultKeyframeCloud (new pcl::PointCloud<pcl::PointXYZI>());
    *resultKeyframeCloud += *cureKeyframeCloud;
    *resultKeyframeCloud += *targetKeyframeCloud;
    *resultKeyframeCloud += *matchKeyframeCloud;
    pcl::io::savePCDFileBinary(DebugDirectory + to_string(_curr_kf_idx) + "_" + to_string(max_eigen) + "_" 
    + to_string(pdop) + ".pcd", *resultKeyframeCloud); // debug data

    if (pdop < dop_thres /*&& max_eigen < 0.005*/)
    {
        Eigen::Matrix3f edge_rot = edge_TF.block(0, 0, 3, 3);
        Eigen::Quaternionf final_q(edge_rot);
        // Get pose transformation
        double roll, pitch, yaw;
        tf::Matrix3x3(tf::Quaternion(final_q.x(), final_q.y(), final_q.z(), final_q.w())).getRPY(roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(edge_TF(0,3), edge_TF(1,3), edge_TF(2,3)));

        double loopNoiseScore = max_eigen; // constant is ok...
        robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
        robustLoopNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(2.0), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

        isLoopClosed = true;
        return poseFrom.between(poseTo);
    }
    else
    {
        return std::nullopt;
    }
}

void process_lcd(void)
{
    float loopClosureFrequency = 0.5; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performSOLiDLoopClosure();
    }
} // process_lcd

void process_edge(void)
{
    while (1)
    {
        while(!solidLoopBuf.empty())
        {
            mLoop.lock();
            std::tuple<int, int, Eigen::Matrix4f> loop_idx_pair = solidLoopBuf.front();
            solidLoopBuf.pop();
            const int prev_node_idx = get<0>(loop_idx_pair);
            const int curr_node_idx = get<1>(loop_idx_pair);
            const Eigen::Matrix4f delta_TF = get<2>(loop_idx_pair);
            auto relative_pose_optional = doGICPVirtualRelative(prev_node_idx, curr_node_idx, delta_TF);
            
            if(relative_pose_optional)
            {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                geometry_msgs::Point p;
                p.x = keyframePoses[prev_node_idx].x;    p.y = keyframePoses[prev_node_idx].y;    p.z = keyframePoses[prev_node_idx].z;
                loopLine.points.push_back(p);
                p.x = keyframePoses[curr_node_idx].x;    p.y = keyframePoses[curr_node_idx].y;    p.z = keyframePoses[curr_node_idx].z;
                loopLine.points.push_back(p);
                LoopLineMarker_pub.publish(loopLine);
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
            }
            mLoop.unlock();
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void process_optimization(void)
{
    float hz = 0.5;
    ros::Rate rate(hz);
    while (ros::ok())
    {
        rate.sleep();
        if(gtSAMgraphMade)
        {
            mBuf.lock();
            runISAM2opt();
            mBuf.unlock();            
        }
    }
}

void process_viz(void)
{
    while(1)
    {
        mViz.lock();
        pcl::PointCloud<pcl::PointXYZI>::Ptr VizMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
        VizMapCloud->points.clear();
        for (int i = 0; i < recentIdxUpdated; i++)
        {   
            double dist = sqrt(pow(keyframePosesUpdated.back().x-keyframePosesUpdated[i].x,2)
                            +pow(keyframePosesUpdated.back().y-keyframePosesUpdated[i].y,2)
                            +pow(keyframePosesUpdated.back().z-keyframePosesUpdated[i].z,2));
            if (dist > 50)  continue;
            Eigen::Matrix3f rotation;
            rotation = Eigen::AngleAxisf(keyframePosesUpdated[i].yaw, Eigen::Vector3f::UnitZ())
                     * Eigen::AngleAxisf(keyframePosesUpdated[i].pitch, Eigen::Vector3f::UnitY())
                     * Eigen::AngleAxisf(keyframePosesUpdated[i].roll, Eigen::Vector3f::UnitX());
            Eigen::Quaternionf q(rotation);        
            Eigen::Matrix4f TF (Eigen::Matrix4f::Identity());
            TF.block(0,0,3,3) = rotation;
            TF(0,3) = keyframePosesUpdated[i].x;
            TF(1,3) = keyframePosesUpdated[i].y;
            TF(2,3) = keyframePosesUpdated[i].z;
            
            pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::io::loadPCDFile(ScansDirectory + std::to_string(i) + ".pcd", *cureKeyframeCloud);
            pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, TF);
            *VizMapCloud += *cureKeyframeCloud; 
        }
        downSizeFilter_map.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilter_map.setInputCloud(VizMapCloud);
        downSizeFilter_map.filter(*VizMapCloud);
        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(*VizMapCloud, map_msg);
        map_msg.header.frame_id = "camera_init";
        PubPGO_map.publish(map_msg);
        mViz.unlock();
        MapCloud->points.clear();
        *MapCloud = *VizMapCloud;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
            
}

void SigHandle(int sig)
{
    ROS_INFO("Saving graph optimization trajectory");
    pcl::PointCloud<PointType>::Ptr OptimizedMapCloud(new pcl::PointCloud<PointType>());
    for (int k = 0; k < keyframePosesUpdated.size(); k++)
    {
        Eigen::Matrix3f rotation;
        rotation = Eigen::AngleAxisf(keyframePosesUpdated[k].yaw, Eigen::Vector3f::UnitZ())
                    * Eigen::AngleAxisf(keyframePosesUpdated[k].pitch, Eigen::Vector3f::UnitY())
                    * Eigen::AngleAxisf(keyframePosesUpdated[k].roll, Eigen::Vector3f::UnitX());
        Eigen::Quaternionf q(rotation);
        optimized_stream << keyframeTimes[k] << " "
                << keyframePosesUpdated[k].x << " " << keyframePosesUpdated[k].y << " " << keyframePosesUpdated[k].z << " " 
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() <<endl;
        
        Eigen::Matrix4f TF (Eigen::Matrix4f::Identity());
        TF.block(0,0,3,3) = rotation;
        TF(0,3) = keyframePosesUpdated[k].x;
        TF(1,3) = keyframePosesUpdated[k].y;
        TF(2,3) = keyframePosesUpdated[k].z;
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::io::loadPCDFile(ScansDirectory + std::to_string(k) + ".pcd", *cureKeyframeCloud);
        pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, TF);
        *OptimizedMapCloud += *cureKeyframeCloud;
    }

    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilter.setInputCloud(OptimizedMapCloud);
    downSizeFilter.filter(*OptimizedMapCloud);
    pcl::io::savePCDFileBinary(save_directory + "OptimizedMap.pcd", *OptimizedMapCloud); 

    ROS_INFO("Optimization trajectory file saved.");
    sig_buffer.notify_all();
    ros::shutdown(); // ROS 종료
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pose_graph_optimization");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    save_directory = string(ROOT_DIR) + "Map/";
    auto unused = system((std::string("exec rm -r ") + save_directory).c_str());
    unused = system((std::string("mkdir -p ") + save_directory).c_str());

    optimized_poseDirectory = save_directory + "optimized_poses.txt";
    odom_poseDirectory = save_directory + "odom_poses.txt";
    odom_stream = std::fstream(odom_poseDirectory, std::fstream::out);
    odom_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!odom_stream) 
    {
        ROS_ERROR("Failed to open odom file");
    }

    optimized_stream = std::fstream(optimized_poseDirectory, std::fstream::out);
    optimized_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!optimized_stream) 
    {
        ROS_ERROR("Failed to open graph optimization file");
    }

    ScansDirectory = save_directory + "Scans/";
    unused = system((std::string("exec rm -r ") + ScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + ScansDirectory).c_str());

    DebugDirectory = save_directory + "Debug/";
    unused = system((std::string("exec rm -r ") + DebugDirectory).c_str());
    unused = system((std::string("mkdir -p ") + DebugDirectory).c_str());


    nh.param<double>("posegraph/r_solid_thres", R_SOLiD_THRES, 0.99);
    nh.param<double>("posegraph/fov_u", FOV_u, 2);
    nh.param<double>("posegraph/fov_d", FOV_d, -24.8);
    nh.param<int>("posegraph/num_angle", NUM_ANGLE, 60);
    nh.param<int>("posegraph/num_range", NUM_RANGE, 40);
    nh.param<int>("posegraph/num_height", NUM_HEIGHT, 32);
    nh.param<int>("posegraph/min_distance", MIN_DISTANCE, 3);
    nh.param<int>("posegraph/max_distance", MAX_DISTANCE, 80);
    nh.param<double>("posegraph/voxel_size", VOXEL_SIZE, 0.4);
    nh.param<int>("posegraph/num_exclude_recent", NUM_EXCLUDE_RECENT, 30);
    nh.param<int>("posegraph/num_candidates_from_tree", NUM_CANDIDATES_FROM_TREE, 3);
    nh.param<double>("posegraph/dop_thres", dop_thres, 0.5);
    solidModule.setParams(FOV_u, FOV_d, NUM_ANGLE, NUM_RANGE, NUM_HEIGHT, MIN_DISTANCE, MAX_DISTANCE, VOXEL_SIZE, NUM_EXCLUDE_RECENT, NUM_CANDIDATES_FROM_TREE, R_SOLiD_THRES);

    gicp.setMaxCorrespondenceDistance(3.0);
    gicp.setNumThreads(0);
    gicp.setCorrespondenceRandomness(15);
    gicp.setMaximumIterations(20);
    gicp.setTransformationEpsilon(0.01);
    gicp.setEuclideanFitnessEpsilon(0.01);
    gicp.setRANSACIterations(5);
    gicp.setRANSACOutlierRejectionThreshold(1.0);

    loopLine.type = visualization_msgs::Marker::LINE_LIST;
    loopLine.action = visualization_msgs::Marker::ADD;
    loopLine.color.b = 1.0; loopLine.color.a = 0.7;
    loopLine.scale.x = 0.1;
    loopLine.header.frame_id = "camera_init";

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    ros::Subscriber sub_keyframe = nh.subscribe("/key_frame", 1, kf_callback);
    kf_node_pub = nh.advertise<sensor_msgs::PointCloud2>("/kf_node", 1);
    LoopLineMarker_pub = nh.advertise<visualization_msgs::Marker>("/loopLine", 1);
    PubPGO_path = nh.advertise<nav_msgs::Path>("/PGO_path", 1);
    PubPGO_map = nh.advertise<sensor_msgs::PointCloud2>("/PGO_map", 1);
    PubScan_range = it.advertise("/scan_range_image", 1);
    PubMap_range = it.advertise("/map_range_image", 1);

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    signal(SIGINT, SigHandle);
    std::thread lc_detection {process_lcd}; // loop closure detection 
    std::thread edge_calculation {process_edge};    //GICP based edge measurement calculation
    std::thread graph_optimization {process_optimization};  //pose graph optimization
    std::thread vis_map {process_viz};  //Map Visualization

    ros::spin();
    odom_stream.close();
    optimized_stream.close();

    return 0;
}