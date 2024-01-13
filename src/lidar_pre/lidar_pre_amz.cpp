// 깃에 추가함.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
// #include "velodyne_filter/msgVelodyne.h"
#include <geometry_msgs/msg/point.hpp>
#include <limits>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// etc
#include <cmath>
#include <vector>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/bool.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/surface/mls.h>

#include <vision_msgs/msg/bounding_box3_d.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include <visualization_msgs/msg/marker_array.hpp>
#include <typeinfo>
#include <cstdio>

using namespace std;
using namespace pcl;
using PointT = pcl::PointXYZI;

std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> clusters;

class VelodynePreprocess : public rclcpp::Node
{
public:
  sensor_msgs::msg::PointCloud2 output;

public:
  VelodynePreprocess()
      : Node("velodyne_preprocess")
  {
    RCLCPP_INFO(this->get_logger(), "Velodyne Preprocess Node has been started");

    LiDAR_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points_filtered", 1);

    Bbox_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
        "/lidar_bbox", 1);

    LiDAR_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", 10,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) -> void
        {
          LiDARCallback(msg);
        });
  }
  ~VelodynePreprocess()
  {
    RCLCPP_INFO(this->get_logger(), "Velodyne Preprocess Node has been terminated");
  }

public:
  void LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr Bbox_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_sub_;
};

void VelodynePreprocess::LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  vision_msgs::msg::Detection3DArray bbox_array;
  bbox_array.header.stamp = msg->header.stamp;

  pcl::PointCloud<PointT>::Ptr cloud_origin(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_down(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_crop(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_cluter(new pcl::PointCloud<PointT>);

  pcl::fromROSMsg(*msg, *cloud_origin);

  // 다운샘플링
  pcl::VoxelGrid<PointT> VG;
  VG.setInputCloud(cloud_origin);
  VG.setLeafSize(0.05f, 0.05f, 0.05f);
  VG.filter(*cloud_down);

  // 크롭필터
  pcl::CropBox<PointT> cropFilter;
  cropFilter.setInputCloud(cloud_down);
  cropFilter.setMin(Eigen::Vector4f(-0.4, -8.0, -0.55, 1.0)); // x,y,z,1
  cropFilter.setMax(Eigen::Vector4f(15.0, 8.0, 0.3, 1.0));
  cropFilter.filter(*cloud_crop);

  // 클러스터링
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud_crop);
  vector<pcl::PointIndices> cluster_indices;

  pcl::EuclideanClusterExtraction<PointT> ECE;
  ECE.setInputCloud(cloud_crop);
  ECE.setClusterTolerance(0.4); // 1m
  ECE.setMinClusterSize(5);
  ECE.setMaxClusterSize(1000);
  ECE.setSearchMethod(tree);
  ECE.extract(cluster_indices);

  int ii = 0;
  pcl::PointCloud<PointT> TotalCloud;
  clusters.clear();
  for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it, ++ii)
  {
    pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
    for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
    {
      cluster->points.push_back(cloud_crop->points[*pit]);
      PointT pt = cloud_crop->points[*pit];
      PointT pt2;
      pt2.x = pt.x, pt2.y = pt.y, pt2.z = pt.z;
      pt2.intensity = (float)(ii + 1);
      TotalCloud.push_back(pt2);
    }
    cluster->width = cluster->size();
    cluster->height = 1;
    cluster->is_dense = true;
    clusters.push_back(cluster);
  }
  pcl::PCLPointCloud2 cloud_p;
  pcl::toPCLPointCloud2(TotalCloud, cloud_p);
  pcl_conversions::fromPCL(cloud_p, output);
  output.header.frame_id = "velodyne";
  output.header.stamp = msg->header.stamp;
  LiDAR_pub_->publish(output);

  // Bounding Box
  // vision_msgs::msg::Detection3DArray bbox_array;
  for (int i = 0; i < clusters.size(); i++)
  {
    Eigen::Vector4f centroid;
    Eigen::Vector4f min_p;
    Eigen::Vector4f max_p;
    Eigen::Vector3f scale_;

    pcl::compute3DCentroid(*clusters[i], centroid);
    pcl::getMinMax3D(*clusters[i], min_p, max_p); // min_p와 max_p는 vector4f이므로 출력은 min_p[0] 이런 식으로 뽑을 수 있다.

    geometry_msgs::msg::Point center_point;
    center_point.x = centroid[0];
    center_point.y = centroid[1];
    center_point.z = centroid[2];

    geometry_msgs::msg::Point min_point;
    min_point.x = min_p.x(); // 이렇게 해도 되네?
    min_point.y = min_p[1];
    min_point.z = min_p[2];

    geometry_msgs::msg::Point max_point;
    max_point.x = max_p[0];
    max_point.y = max_p[1];
    max_point.z = max_p[2];

    float width = max_p[0] - min_p[0];
    float height = max_p[1] - min_p[1];
    float depth = max_p[2] - min_p[2];

    Eigen::Quaternionf rotation(0.0, 0.0, 0.0, 1.0);
    geometry_msgs::msg::Quaternion quaternion;
    quaternion.x = rotation.x();
    quaternion.y = rotation.y();
    quaternion.z = rotation.z();
    quaternion.w = rotation.w();

    vision_msgs::msg::Detection3D detection;
    vision_msgs::msg::BoundingBox3D bbox;

    bbox.center.position = center_point;
    bbox.center.orientation = quaternion;
    bbox.size.x = width;
    bbox.size.y = height;
    bbox.size.z = depth;

    detection.header.frame_id = "velodyne";
    detection.header.stamp = msg->header.stamp;
    detection.bbox = bbox;

    bbox_array.detections.push_back(detection);
  }
  // bbox_array.header.stamp = msg->header.stamp;
  this->Bbox_pub_->publish(bbox_array);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelodynePreprocess>());
  rclcpp::shutdown();
  return 0;
}
