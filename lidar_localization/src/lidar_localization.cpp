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
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/Vector3.h>

#include <nano_gicp/point_type_nano_gicp.hpp>
#include <nano_gicp/nano_gicp.hpp>

using namespace std;

string map_directory;
string lidar_topic, odom_topic, imu_topic;
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudMap(new pcl::PointCloud<pcl::PointXYZI>());

nano_gicp::NanoGICP<PointType2, PointType2> gicp;
pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
double voxel_size = 0;
double corres_dist, epsilon, euclidean_epsilon, ransac_outlier;
double std_x, std_y, std_z;
double std_roll, std_pitch, std_yaw;
int num_thread, corres_random, iter_num, ransac_iter;
double dt, curr_odom_time, prev_odom_time;
double imu_dt, curr_imu_time, prev_imu_time;
bool imu_flag = false;
bool odom_flag = false;
double linear_velo = 0;
std::vector<Eigen::Vector3d> biasVec;
Eigen::Vector3d gyro_bias;
int imu_cnt = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree (new pcl::KdTreeFLANN<pcl::PointXYZI>());
std::vector<int> indiceLet;
std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;
Eigen::VectorXd state(6);
Eigen::VectorXd state_inc(6);
Eigen::VectorXd state_time_inc(6);
Eigen::VectorXd prev_TP(6);
nav_msgs::Path local_path; 

//KF Param
Eigen::VectorXd measurement(6);
Eigen::VectorXd residual(6);
Eigen::MatrixXd Q(6,6);
Eigen::MatrixXd H(6,6);
Eigen::MatrixXd P(6,6);
Eigen::MatrixXd R(6,6);
Eigen::MatrixXd K(6,6);

ros::Publisher pubMapCloud, pubMatchingResult, pubBodyCloud, pubOdom, pubPath;

double pi2pi(double radian)
{
    if (radian > M_PI)  radian -= 2 * M_PI;
    else if (radian < -M_PI)    radian += 2 * M_PI;

    return radian;
}
double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

double deg2rad(double degree)
{
  return degree * M_PI / 180.0;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr pc_undistortion(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr undistort_cloud (new pcl::PointCloud<pcl::PointXYZI>());
    int cloudSize = input_cloud->points.size();
    pcl::PointXYZI point;

    float startOri = -atan2(input_cloud->points[0].y, input_cloud->points[0].x);
    float endOri = -atan2(input_cloud->points[cloudSize - 1].y,
                          input_cloud->points[cloudSize - 1].x) +
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
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = input_cloud->points[i].x;
        point.y = input_cloud->points[i].y;
        point.z = input_cloud->points[i].z;
        point.intensity = input_cloud->points[i].intensity;

        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        {
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        float relTime = (ori - startOri) / (endOri - startOri);
        Eigen::Vector3f linearInc;
        Eigen::Vector3f angInc;
        linearInc(0) = state_inc(0) * relTime;
        linearInc(1) = state_inc(1) * relTime;
        linearInc(2) = state_inc(2) * relTime;
        angInc(0) = state_inc(3) * relTime;
        angInc(1) = state_inc(4) * relTime;
        angInc(2) = state_inc(5) * relTime;
       
        Eigen::Matrix3f R;
        R = Eigen::AngleAxisf(angInc(2), Eigen::Vector3f::UnitZ())
          * Eigen::AngleAxisf(angInc(1), Eigen::Vector3f::UnitY())
          * Eigen::AngleAxisf(angInc(0), Eigen::Vector3f::UnitX());
        pcl::PointXYZ tempPoint;
        tempPoint.x = R(0,0) * point.x + R(0,1) * point.y + R(0,2) * point.z + linearInc(0);
        tempPoint.y = R(1,0) * point.x + R(1,1) * point.y + R(1,2) * point.z + linearInc(1);
        tempPoint.z = R(2,0) * point.x + R(2,1) * point.y + R(2,2) * point.z + linearInc(2);
        point.x = tempPoint.x;  point.y = tempPoint.y;  point.z = tempPoint.z;
        undistort_cloud->points.push_back(point);
    }
    state_inc.setZero();
    return undistort_cloud;
}

void imu_callback(const sensor_msgs::Imu::ConstPtr &imu) 
{
    if (!imu_flag) 
    {
        imu_flag = true;
        prev_imu_time = imu->header.stamp.toSec();
        return;
    }
    curr_odom_time = imu->header.stamp.toSec();
    imu_dt = curr_odom_time - prev_imu_time;

    if (linear_velo < 0.01 && fabs(rad2deg(imu->angular_velocity.z)) < 0.1) //로봇 정지 시 바이어스 벡터 push back
    {
        Eigen::Vector3d temp_bias;
        temp_bias(0) = imu->angular_velocity.x;
        temp_bias(1) = imu->angular_velocity.y;
        temp_bias(2) = imu->angular_velocity.z;
        biasVec.push_back(temp_bias);
    }
    if (biasVec.size() > 100)  //100개 이상 시 이전 데이터 삭제
    {
        biasVec.erase(biasVec.begin());
    }
    gyro_bias(0) = 0;
    gyro_bias(1) = 0;
    gyro_bias(2) = 0;
    //RPY 바이어스 계산
    for (size_t k = 0; k < biasVec.size(); k++)
    {
        gyro_bias(0) += biasVec[k](0);
        gyro_bias(1) += biasVec[k](1);
        gyro_bias(2) += biasVec[k](2);
    }
    gyro_bias(0) /= biasVec.size();
    gyro_bias(1) /= biasVec.size();
    gyro_bias(2) /= biasVec.size(); 

    Eigen::Vector3d gyro;
    gyro(0) = imu->angular_velocity.x - gyro_bias(0);
    gyro(1) = imu->angular_velocity.y - gyro_bias(1);
    gyro(2) = imu->angular_velocity.z - gyro_bias(2);
    
    Q.diagonal() << 0, 0, 0, pow(deg2rad(0.1), 2), pow(deg2rad(0.1), 2), pow(deg2rad(0.1), 2);
    P += Q;

    if (linear_velo > 0.01 || std::fabs(rad2deg(imu->angular_velocity.z)) > deg2rad(0.5))
    {     
        state(3) += gyro(0)*imu_dt; state(4) += gyro(1)*imu_dt; state(5) += gyro(2)*imu_dt;
        state_inc(3) += gyro(0)*imu_dt; state_inc(4) += gyro(1)*imu_dt; state_inc(5) += gyro(2)*imu_dt;
    }
    prev_imu_time = curr_odom_time;
}
void odom_callback(const nav_msgs::Odometry::ConstPtr &odom) 
{
    if (!odom_flag) 
    {
        odom_flag = true;
        prev_odom_time = odom->header.stamp.toSec();
        return;
    }
    linear_velo = odom->twist.twist.linear.x;
    double angular_velo = odom->twist.twist.angular.z;
    curr_odom_time = odom->header.stamp.toSec();
    dt = curr_odom_time - prev_odom_time;
    
    Q.diagonal() << pow(0.01, 2), pow(0.01, 2), 0, 0, 0, 0;
    P += Q;

    if (linear_velo > 0.01 || std::fabs(angular_velo) > deg2rad(0.5))
    {     
        // state(5) += angular_velo*dt;
        // pi2pi(state(5));
        state(0) += linear_velo*cos(state(5)) * dt; state(1) += linear_velo*sin(state(5)) * dt;
        
        // state_inc(5) += angular_velo*dt;
        state_inc(0) += linear_velo*cos(state(5)) * dt; state_inc(1) += linear_velo*sin(state(5)) * dt;
    }

    Eigen::Vector3d diff_TP;
    diff_TP(0) = linear_velo*cos(state(5)) * dt - prev_TP(0);
    diff_TP(1) = linear_velo*sin(state(5)) * dt - prev_TP(1);
    diff_TP(2) = deg2rad(angular_velo*dt) - prev_TP(2);
    diff_TP(2) = pi2pi(diff_TP(2));
    prev_TP(0) = linear_velo*cos(state(5)) * dt;
    prev_TP(1) = linear_velo*sin(state(5)) * dt;
    prev_TP(2) = deg2rad(angular_velo*dt);
    prev_TP(2) = pi2pi(prev_TP(2));
    state_time_inc(0) += diff_TP(0);
    state_time_inc(1) += diff_TP(1);
    state_time_inc(5) += diff_TP(2);

    Eigen::Matrix3f R;
    R =  Eigen::AngleAxisf(state(5), Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(state(4), Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(state(3), Eigen::Vector3f::UnitX());
    Eigen::Quaternionf quat(R); 
    geometry_msgs::Quaternion q_msg;
    q_msg.x = quat.x();
    q_msg.y = quat.y();
    q_msg.z = quat.z();
    q_msg.w = quat.w(); 

    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time().fromSec(curr_odom_time);
    odom_msg.header.frame_id = "map";
    odom_msg.child_frame_id = "odom";
    odom_msg.pose.pose.position.x = state(0);
    odom_msg.pose.pose.position.y = state(1);
    odom_msg.pose.pose.position.z = state(2);
    odom_msg.pose.pose.orientation.w = q_msg.w;
    odom_msg.pose.pose.orientation.x = q_msg.x;
    odom_msg.pose.pose.orientation.y = q_msg.y;
    odom_msg.pose.pose.orientation.z = q_msg.z;
    for (int i = 0; i < 6; i ++)
    {
        odom_msg.pose.covariance[i*6 + 0] = P(i, 0);
        odom_msg.pose.covariance[i*6 + 1] = P(i, 1);
        odom_msg.pose.covariance[i*6 + 2] = P(i, 2);
        odom_msg.pose.covariance[i*6 + 3] = P(i, 3);
        odom_msg.pose.covariance[i*6 + 4] = P(i, 4);
        odom_msg.pose.covariance[i*6 + 5] = P(i, 5);
    }

    pubOdom.publish(odom_msg);
    geometry_msgs::PoseStamped local_pose;
    local_pose.pose.position.x = state(0);
    local_pose.pose.position.y = state(1);
    local_pose.pose.position.z = state(2);
    local_pose.pose.orientation.w = q_msg.w;
    local_pose.pose.orientation.x = q_msg.x;
    local_pose.pose.orientation.y = q_msg.y;
    local_pose.pose.orientation.z = q_msg.z;

    local_path.header.stamp = ros::Time().fromSec(curr_odom_time);
    local_path.header.frame_id = "map";

    local_path.poses.push_back(local_pose);
    while (local_path.poses.size() > 1000)
    {
        local_path.poses.erase(local_path.poses.begin());
    }
    pubPath.publish(local_path);    

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odom_msg.pose.pose.position.x, \
                                    odom_msg.pose.pose.position.y, \
                                    odom_msg.pose.pose.position.z));
    q.setW(odom_msg.pose.pose.orientation.w);
    q.setX(odom_msg.pose.pose.orientation.x);
    q.setY(odom_msg.pose.pose.orientation.y);
    q.setZ(odom_msg.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odom_msg.header.stamp, "map", "odom" ) );

    prev_odom_time = curr_odom_time;
}

void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &lidar) 
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr curr_PointCloud (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr undistort_PointCloud (new pcl::PointCloud<pcl::PointXYZI>());
    double lidar_time = lidar->header.stamp.toSec();
    pcl::fromROSMsg(*lidar, *curr_PointCloud);
    undistort_PointCloud = pc_undistortion(curr_PointCloud);

    downSizeFilter.setInputCloud(undistort_PointCloud);
    downSizeFilter.filter(*undistort_PointCloud);

    gicp.setInputSource(undistort_PointCloud);
    
    Eigen::Matrix4f init_TF(Eigen::Matrix4f::Identity());
    Eigen::Matrix3f Rotation;
    Rotation =  Eigen::AngleAxisf(state(5), Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf(state(4), Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(state(3), Eigen::Vector3f::UnitX());
    init_TF.block(0,0,3,3) = Rotation;
    init_TF(0,3) = state(0);
    init_TF(1,3) = state(1);
    init_TF(2,3) = state(2);
    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    gicp.align(*aligned_cloud, init_TF);
    Eigen::Matrix4f match_TF = gicp.getFinalTransformation();
    pcl::PointCloud<pcl::PointXYZI>::Ptr matchedCloud (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*undistort_PointCloud, *matchedCloud, match_TF);

    Eigen::Matrix<double, 6, 6> hessian = gicp.getHessian();
    
    typedef Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> EigenSolver;
    EigenSolver es;
    Eigen::Matrix<double, 6, 6> hessian_inv = hessian.inverse();
    es.compute(hessian_inv);
    
    Eigen::VectorXd eigenvalues = es.eigenvalues().real();
    Eigen::VectorXd sqrt_eigenvalues = eigenvalues.array().sqrt();

    pcl::PointCloud<pcl::PointXYZI>::Ptr MMCloud (new pcl::PointCloud<pcl::PointXYZI>());
    for (int k = 0; k < matchedCloud->points.size(); k++)
    {
        kdtree->nearestKSearch(matchedCloud->points[k], 1, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < 0.1)
        {
            MMCloud->points.push_back(matchedCloud->points[k]);
        }
    }

    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setInputCloud(MMCloud);
    downSizeFilter.setLeafSize(2.0, 2.0, 2.0);
    downSizeFilter.setMinimumPointsNumberPerVoxel(2);
    downSizeFilter.filter(*MMCloud);
    pcl::removeNaNFromPointCloud(*MMCloud, *MMCloud, indiceLet);
    indiceLet.clear();

    std::vector<Eigen::Vector3d> range_info;
    for (size_t k = 0; k < MMCloud->points.size(); k++)
    {
        double r = sqrt(pow((MMCloud->points[k].x - state(0)), 2) + pow((MMCloud->points[k].y - state(1)), 2) + pow((MMCloud->points[k].z - state(2)), 2));
        if (r < 2.0)
        {
            continue;
        }
        Eigen::Vector3d r_info;
        r_info(0) = (MMCloud->points[k].x - state(0)) / r;
        r_info(1) = (MMCloud->points[k].y - state(1)) / r;
        r_info(2) = (MMCloud->points[k].z - state(2)) / r;
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
    double dop_scale = 2;
    double c = 4;
    double r = pdop;
    //tukey loss function
    double scale = pow(c,2)*(1-pow((1-pow((r/c),2)),3))*dop_scale; 

    if (pdop >= c)  scale = pow(c,2)*dop_scale;       
    // cout<<"\033[1;32m"<<"Eigen : "<<sqrt_eigenvalues.transpose()<< "\033[0m"<<endl;
    // cout<<"\033[1;33m"<<"DOP : "<<pdop<<", scale : "<<scale<< "\033[0m"<<endl;

    std_x = sqrt_eigenvalues(0) * 10;
    std_y = sqrt_eigenvalues(1) * 10;
    std_z = sqrt_eigenvalues(2) * 10;
    std_roll = sqrt_eigenvalues(3) * 10;
    std_pitch = sqrt_eigenvalues(4) * 10;;
    std_yaw = sqrt_eigenvalues(5) *10;

    Eigen::Matrix3f meas_R = match_TF.block(0,0,3,3);

    Eigen::VectorXd z(6);
    Eigen::Quaternionf q_meas(meas_R);
    tf2::Quaternion q_meas2;
    q_meas2.setW(q_meas.w());
    q_meas2.setX(q_meas.x());
    q_meas2.setY(q_meas.y());
    q_meas2.setZ(q_meas.z());
    tf2::Matrix3x3 mat(q_meas2);
    mat.getRPY(z(3), z(4), z(5));
    z(5) = pi2pi(z(5));
    z(0) = match_TF(0,3);
    z(1) = match_TF(1,3);
    z(2) = match_TF(2,3);
    residual = z - H * state;
    residual(5) = pi2pi(residual(5));

    R.diagonal() << pow(scale*std_x, 2), pow(scale*std_y, 2), pow(scale*std_z, 2), pow((scale*std_roll), 2), pow((scale*std_pitch), 2), pow((scale*std_yaw), 2);

    Eigen::MatrixXd HPHTR(6, 6);
    HPHTR = H * P * H.transpose() + R;
    K = P * H.transpose() * HPHTR.inverse();
    state += K * residual;
    state(5) = pi2pi(state(5));
    Eigen::MatrixXd I(6, 6);
    I.setIdentity();
    P = (I - K * H) * P;
    state += state_time_inc;
    state(5) = pi2pi(state(5));   
    state_time_inc.setZero();
    
    sensor_msgs::PointCloud2 lidarMap_msg;
    pcl::toROSMsg(*laserCloudMap, lidarMap_msg);
    lidarMap_msg.header.stamp = ros::Time().fromSec(lidar_time);
    lidarMap_msg.header.frame_id = "map";
    pubMapCloud.publish(lidarMap_msg);

    Eigen::Matrix4f global_TF (Eigen::Matrix4f::Identity());
    Rotation =  Eigen::AngleAxisf(state(5), Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf(state(4), Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(state(3), Eigen::Vector3f::UnitX());
    global_TF.block(0,0,3,3) = Rotation;
    global_TF(0,3) = state(0);
    global_TF(1,3) = state(1);
    global_TF(2,3) = state(2);
    pcl::PointCloud<pcl::PointXYZI>::Ptr globalCloud (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*undistort_PointCloud, *globalCloud, global_TF);

    sensor_msgs::PointCloud2 lidarReg_msg;
    pcl::toROSMsg(*globalCloud, lidarReg_msg);
    lidarReg_msg.header.stamp = ros::Time().fromSec(lidar_time);
    lidarReg_msg.header.frame_id = "map";
    pubMatchingResult.publish(lidarReg_msg);

    sensor_msgs::PointCloud2 lidarBody_msg;
    pcl::toROSMsg(*undistort_PointCloud, lidarBody_msg);
    lidarBody_msg.header.stamp = ros::Time().fromSec(lidar_time);
    lidarBody_msg.header.frame_id = "odom";
    pubBodyCloud.publish(lidarBody_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_localization");
    ros::NodeHandle nh;
    map_directory = string(ROOT_DIR) + "Map/OptimizedMap.pcd";

    if (pcl::io::loadPCDFile(map_directory, *laserCloudMap) == -1)
    {
        ROS_ERROR("Map file cannot open.\n");
        return 0;
    }

    nh.param<string>("common/lidar_topic", lidar_topic, "/velodyne_points");
    nh.param<string>("common/odom_topic", odom_topic, "/scout_base_controller/odom");
    nh.param<string>("common/imu_topic", imu_topic, "/imu/data");
    nh.param<double>("common/voxel_size", voxel_size, 0.1);
    nh.param<double>("gicp/corres_dist", corres_dist, 2.0);
    nh.param<int>("gicp/num_thread", num_thread, 0);
    nh.param<int>("gicp/corres_random", corres_random, 15);
    nh.param<int>("gicp/iter_num", iter_num, 5);
    nh.param<double>("gicp/epsilon", epsilon, 0.01);
    nh.param<double>("gicp/euclidean_epsilon", euclidean_epsilon, 0.01);
    nh.param<int>("gicp/ransac_iter", ransac_iter, 5);
    nh.param<double>("gicp/ransac_outlier", ransac_outlier, 1.0);
    nh.param<double>("localization/init_x", state(0), 0.0);
    nh.param<double>("localization/init_y", state(1), 0.0);
    nh.param<double>("localization/init_z", state(2), 0.0);
    nh.param<double>("localization/init_roll", state(3), 0.0);
    nh.param<double>("localization/init_pitch", state(4), 0.0);
    nh.param<double>("localization/init_yaw", state(5), 0.0);
    
    deg2rad(state(3));
    deg2rad(state(4));
    deg2rad(state(5));

    gicp.setMaxCorrespondenceDistance(corres_dist);
    gicp.setNumThreads(num_thread);
    gicp.setCorrespondenceRandomness(corres_random);
    gicp.setMaximumIterations(iter_num);
    gicp.setTransformationEpsilon(epsilon);
    gicp.setEuclideanFitnessEpsilon(euclidean_epsilon);
    gicp.setRANSACIterations(ransac_iter);
    gicp.setRANSACOutlierRejectionThreshold(ransac_outlier);
    gicp.setInputTarget(laserCloudMap);

    downSizeFilter.setLeafSize(voxel_size, voxel_size, voxel_size);
    kdtree->setInputCloud(laserCloudMap);

    P.diagonal() << 10000, 10000, 10000, 10000, 10000, 10000;
    state.setZero();
    state_inc.setZero();
    state_time_inc.setZero();
    residual.setZero();
    H.setZero();
    H(0, 0) = 1;    H(1, 1) = 1;    H(2, 2) = 1;    H(3, 3) = 1;    H(4, 4) = 1;    H(5, 5) = 1;
    std_x = 0.05;    std_y = 0.05;    std_z = 0.05;    
    std_roll = deg2rad(0.5);    std_pitch = deg2rad(0.5);    std_yaw = deg2rad(0.5);

    ros::Subscriber lidar_sub = nh.subscribe(lidar_topic, 10, lidar_callback);
    ros::Subscriber odom_sub = nh.subscribe(odom_topic, 10, odom_callback);
    ros::Subscriber imu_sub = nh.subscribe(imu_topic, 10, imu_callback);
    pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_map", 1);
    pubMatchingResult = nh.advertise<sensor_msgs::PointCloud2>("/matchingPoints", 1);
    pubBodyCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_body", 1);
    pubOdom = nh.advertise<nav_msgs::Odometry>("/local_result", 1);
    pubPath = nh.advertise<nav_msgs::Path>("/local_path", 1);
    ros::spin(); 

    return 0;
}