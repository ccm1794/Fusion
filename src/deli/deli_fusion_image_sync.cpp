// yolo박스랑 클러스터 박스를 가져와서 서로 비교 후 많이 겹치는 박스끼리 매칭해서 pub
// float32multiarray로 class+위치

#include <rclcpp/rclcpp.hpp>
#include <mutex>
#include <memory>
#include <thread>
#include <pthread.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>

#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <image_transport/image_transport.hpp> //꼭 있을 필요는 없을 듯?
#include <cv_bridge/cv_bridge.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//yolo header 추가하기
#include <std_msgs/msg/int16.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/bounding_box3_d.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// time sync
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "message_filters/subscriber.h"

using namespace std;
using namespace cv;

// 클러스터 박스 좌표를 저장하기 위한 구조체
struct Box 
{
  float x;
  float y;
  float z;

  float size_x;
  float size_y;
  float size_z;
};

// 욜로 박스 저장할 구조체
struct Box_yolo
{
  float x1;
  float x2;
  float y1;
  float y2;
  int color;
};

struct coord_final
{
  float x;
  float y;
  int id;
};

struct Point3D {
    float x;
    float y;
    float z;
};

struct Box_points {
    Point3D points[8];
};  



Box_yolo transformBox(const cv::Mat& transformMat, const Box_points& real_points, float offsetY = 0.0f) {
    std::vector<float> xs(8), ys(8);
    
    for (int i = 0; i < 8; ++i) {
        double box[4] = {real_points.points[i].x, real_points.points[i].y, real_points.points[i].z, 1.0};
        cv::Mat pos(4, 1, CV_64F, box); // 3차원 좌표

        cv::Mat newPos(transformMat * pos); // 카메라 좌표로 변환한 것.
        
        xs[i] = (float)(newPos.at<double>(0, 0) / newPos.at<double>(2, 0));
        ys[i] = (float)(newPos.at<double>(1, 0) / newPos.at<double>(2, 0)) + offsetY;
    }

    return { 
      *std::min_element(xs.begin(), xs.end()), 
      *std::max_element(xs.begin(), xs.end()), 
      *std::min_element(ys.begin(), ys.end()), 
      *std::max_element(ys.begin(), ys.end()) };
}

// 클러스터 박스의 8개의 꼭지점을 구하는 함수
Box_points calcBox_points(const Box &box)
{
  Box_points real_points;

  real_points.points[0].x = box.x - (box.size_x)/2;
  real_points.points[0].y = box.y + (box.size_y)/2;
  real_points.points[0].z = box.z - (box.size_z)/2;

  real_points.points[1].x = box.x - (box.size_x)/2;
  real_points.points[1].y = box.y - (box.size_y)/2;
  real_points.points[1].z = box.z - (box.size_z)/2;

  real_points.points[2].x = box.x - (box.size_x)/2;
  real_points.points[2].y = box.y - (box.size_y)/2;
  real_points.points[2].z = box.z + (box.size_z)/2;

  real_points.points[3].x = box.x - (box.size_x)/2;
  real_points.points[3].y = box.y + (box.size_y)/2;
  real_points.points[3].z = box.z + (box.size_z)/2;

  real_points.points[4].x = box.x + (box.size_x)/2;
  real_points.points[4].y = box.y + (box.size_y)/2;
  real_points.points[4].z = box.z - (box.size_z)/2;

  real_points.points[5].x = box.x + (box.size_x)/2;
  real_points.points[5].y = box.y - (box.size_y)/2;
  real_points.points[5].z = box.z - (box.size_z)/2;

  real_points.points[6].x = box.x + (box.size_x)/2;
  real_points.points[6].y = box.y - (box.size_y)/2;
  real_points.points[6].z = box.z + (box.size_z)/2;

  real_points.points[7].x = box.x + (box.size_x)/2;
  real_points.points[7].y = box.y + (box.size_y)/2;
  real_points.points[7].z = box.z + (box.size_z)/2;

  return real_points;
}

// iou 계산
float get_iou(const Box_yolo &a, const Box_yolo &b)
{
  float x_left = max(a.x1, b.x1);
  float y_top = max(a.y1, b.y1);
  float x_right = min(a.x2, b.x2);
  float y_bottom = min(a.y2, b.y2);

  if (x_right < x_left || y_bottom < y_top)
    return 0.0f;
  
  float intersection_area = (x_right - x_left) * (y_bottom - y_top);

  float area1 = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area2 = (b.x2 - b.x1) * (b.y2 - b.y1);

  float iou = intersection_area / (area1 + area2 - intersection_area);

  return iou;
}

std::mutex mut_box, mut_yolo;
std::vector<Box> boxes;
std::vector<Box_yolo> boxes_yolo;

using Stamped3PointMsg = vision_msgs::msg::Detection3DArray;
using Stamped3PointMsgSubscriber = message_filters::Subscriber<Stamped3PointMsg>;

using Stamped2PointMsg = vision_msgs::msg::Detection2DArray;
using Stamped2PointMsgSubscriber = message_filters::Subscriber<Stamped2PointMsg>;

using ApproximateSyncPolicy = message_filters::sync_policies::ApproximateTime<Stamped3PointMsg, Stamped2PointMsg>;
using ApproximateSync = message_filters::Synchronizer<ApproximateSyncPolicy>;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  Mat transformMat;
  Mat CameraMat;
  Mat DistCoeff;

  int cluster_count;

  int box_locker = 0; // box_Callback 할 때 lock해주는 변수
  int yolo_locker = 0;
  int deli_flag = 1;
  int sync_locker = 0;

  vector<double> CameraExtrinsic_vector;
  vector<double> CameraMat_vector;
  vector<double> DistCoeff_vector;

public:
  ImageLiDARFusion()
  : Node("fusion_deli_sync")
  {
    RCLCPP_INFO(this->get_logger(), "------------initialized------------\n");

    // 파라미터 선언 영역
    this->declare_parameter("CameraExtrinsicMat", vector<double>());
    this->CameraExtrinsic_vector = this->get_parameter("CameraExtrinsicMat").as_double_array();
    this->declare_parameter("CameraMat", vector<double>());
    this->CameraMat_vector = this->get_parameter("CameraMat").as_double_array();
    this->declare_parameter("DistCoeff", vector<double>());
    this->DistCoeff_vector = this->get_parameter("DistCoeff").as_double_array();    

    // 받아온 파라미터로 변환행렬 계산
    this->set_param();

    // message filters
    rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_default;
    custom_qos_profile.depth = 3;

    yolo_detect_sub_ = std::make_shared<Stamped2PointMsgSubscriber>(this, "/yolo_detect", rmw_qos_profile_default);
    lidar_box_sub_ = std::make_shared<Stamped3PointMsgSubscriber>(this, "/lidar_bbox", rmw_qos_profile_default);

    approximate_sync_ = std::make_shared<ApproximateSync>(ApproximateSyncPolicy(5), *lidar_box_sub_, *yolo_detect_sub_);
    approximate_sync_->registerCallback(std::bind(&ImageLiDARFusion::approximateSyncCallback, this, std::placeholders::_1, std::placeholders::_2));

    publish_point_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("coord_xyz", 1);
    publish_cone_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/coord_xy", 1);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/marker", 10);

    mission_sub_ = this->create_subscription<std_msgs::msg::Int16>(
      "/Planning/deli_flag", 1,
      [this](const std_msgs::msg::Int16::SharedPtr msg) -> void
      {
        this->deli_flag = msg->data;
      });

    // 타이머 콜백으로 박스 매칭함수 실행
    auto timer_callback = [this]() -> void {FusionCallback();};
    timer_ = create_wall_timer(50ms, timer_callback); 
    
    RCLCPP_INFO(this->get_logger(), "------------initialize end------------\n");
  }

  ~ImageLiDARFusion()
  {
  }

public:
  void set_param();
  void FusionCallback();

private:
  std::shared_ptr<Stamped3PointMsgSubscriber> lidar_box_sub_;
  std::shared_ptr<Stamped2PointMsgSubscriber> yolo_detect_sub_;
  std::shared_ptr<ApproximateSync> approximate_sync_;

  rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr mission_sub_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publish_cone_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publish_point_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  void approximateSyncCallback(
    const std::shared_ptr<const vision_msgs::msg::Detection3DArray>& lidar_msg,
    const std::shared_ptr<const vision_msgs::msg::Detection2DArray>& yolo_msg
    );
  
};

// 파라미터 설정
void ImageLiDARFusion::set_param()
{
  Mat CameraExtrinsicMat;
  Mat concatedMat;

  Mat CameraExtrinsicMat_(4,4, CV_64F, CameraExtrinsic_vector.data());
  Mat CameraMat_(3,3, CV_64F, CameraMat_vector.data());
  Mat DistCoeffMat_(1,4, CV_64F, DistCoeff_vector.data());

  //위에 있는 내용 복사
  CameraExtrinsicMat_.copyTo(CameraExtrinsicMat);
  CameraMat_.copyTo(this->CameraMat);
  DistCoeffMat_.copyTo(this->DistCoeff);

  //재가공 : 회전변환행렬, 평행이동행렬
  Mat Rlc = CameraExtrinsicMat(cv::Rect(0,0,3,3));
  Mat Tlc = CameraExtrinsicMat(cv::Rect(3,0,1,3));

  cv::hconcat(Rlc, Tlc, concatedMat);

  this->transformMat = this->CameraMat * concatedMat;

  RCLCPP_INFO(this->get_logger(), "transform Matrix ready");
}

// sync callback
void ImageLiDARFusion::approximateSyncCallback(
  const std::shared_ptr<const vision_msgs::msg::Detection3DArray>& lidar_msg,
    const std::shared_ptr<const vision_msgs::msg::Detection2DArray>& yolo_msg
)
{
  if(this->sync_locker == 0 && this->deli_flag == 1)
  {
    if(lidar_msg->detections.size() != 0)
    {
      for(int i = 0; i < lidar_msg->detections.size(); i++)
      { 
        if(lidar_msg->detections[i].bbox.size.y < 1 && lidar_msg->detections[i].bbox.size.z < 2)
        {
          Box box = 
          {
            lidar_msg->detections[i].bbox.center.position.x, 
            lidar_msg->detections[i].bbox.center.position.y, 
            lidar_msg->detections[i].bbox.center.position.z, 
            lidar_msg->detections[i].bbox.size.x, 
            lidar_msg->detections[i].bbox.size.y,  // 투영되는 박스 크기를 임의로 변형
            lidar_msg->detections[i].bbox.size.z // 투영되는 박스 크기를 임의로 변형
          };
          boxes.push_back(box);
        }
      }
    }

    if (yolo_msg->detections.size() != 0)
    {
      std::string Class_name;

      for (int i = 0; i <yolo_msg->detections.size(); i++)
      {
        Class_name = yolo_msg->detections[i].results[0].id; // 클래스 가져오기

        int color = 0;

        if(Class_name == "A1")
        {
          color = 1;
        }
        else if(Class_name == "A2")
        {
          color = 2;
        }
        else if(Class_name == "A3")
        {
          color = 3;
        }
        else if(Class_name == "B1")
        {
          color = 4;
        }      
        else if(Class_name == "B2")
        {
          color = 5;
        } 
        else if(Class_name == "B3")
        {
          color = 6;
        } 
        else
        {
          color = 0;
        }
        if(yolo_msg->detections[i].bbox.size_x / yolo_msg->detections[i].bbox.size_y > 0.4)
        {
          Box_yolo box_yolo = 
          {
            yolo_msg->detections[i].bbox.center.x - (yolo_msg->detections[i].bbox.size_x)/2, // x1
            yolo_msg->detections[i].bbox.center.x + (yolo_msg->detections[i].bbox.size_x)/2, // x2
            yolo_msg->detections[i].bbox.center.y - (yolo_msg->detections[i].bbox.size_y)/2, // y1
            yolo_msg->detections[i].bbox.center.y + (yolo_msg->detections[i].bbox.size_y)/2, // y2
            color
          };
          boxes_yolo.push_back(box_yolo);
        }
      }
    }
    this->sync_locker = 1;
  }
}

// fusion callback
void ImageLiDARFusion::FusionCallback()
{
  if(this->sync_locker == 1 && this->deli_flag == 1)
  {
    std::vector<Box> lidar_boxes = boxes;
    std::vector<Box_yolo> yolo_boxes = boxes_yolo;
    
    // 투영 위한 검정 이미지
    cv::Mat image(960, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    if (this->yolo_locker == 1 && this->box_locker == 1)
    {
      std::vector<Box_yolo> boxes_2d_cluster;
      for (const auto& Box : lidar_boxes)
      {
        Box_points real_points = calcBox_points(Box);
        boxes_2d_cluster.push_back(transformBox(this->transformMat, real_points));
      }

      std_msgs::msg::Float64MultiArray coord;
      sensor_msgs::msg::PointCloud2 cloud_msg;

      pcl::PointCloud<pcl::PointXYZ>::Ptr coord_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      visualization_msgs::msg::MarkerArray text_array;

      for ( int i = 0; i < yolo_boxes.size(); i++)
      {
        vector<coord_final> coord_vector;
        float max_iou = 0.0f;
        float standard_iou = 0.2;
        int class_id = yolo_boxes[i].color;
        int max_index = -1;

        for (int j = 0; j < boxes_2d_cluster.size(); j++)
        {
          float iou = get_iou(yolo_boxes[i], boxes_2d_cluster[j]);

          if(iou > standard_iou) // 특정 iou를 넘는 박스들을 모두 저장
          {
            coord_final coord_ =
            {
              lidar_boxes[j].x,
              lidar_boxes[j].y,
              class_id
            };
            coord_vector.push_back(coord_);
          }
        }

        if(coord_vector.size() == 0) // 이번 루프에 해당하는 yolo_box와 매칭되는 클러스터가 없다는 뜻.
        {
          continue; // 이번 루프가 끝나고 다음 yolo_box와 클러스터를 비교한다.
        }
        else if(coord_vector.size() == 1)
        {
          // float data
          coord.data.push_back(coord_vector[0].x);
          coord.data.push_back(coord_vector[0].y);
          coord.data.push_back(coord_vector[0].id);
          // pcl data
          pcl::PointXYZ point;
          point.x = coord_vector[0].x;
          point.y = coord_vector[0].y;
          point.z = 0.0f;
          coord_cloud->push_back(point);
          // marker data
          visualization_msgs::msg::Marker text;
          text.header.frame_id = "velodyne";
          text.ns = "text";
          if (class_id == 1){text.text = "A1"; text.color.r = 1.0; text.color.g = 0.0; text.color.b = 0.0;}
          else if (class_id == 2){text.text = "A2"; text.color.r = 0.0; text.color.g = 1.0; text.color.b = 0.0;}
          else if (class_id == 3){text.text = "A3"; text.color.r = 0.0; text.color.g = 0.0; text.color.b = 1.0;}
          else if (class_id == 4){text.text = "B1"; text.color.r = 1.0; text.color.g = 1.0; text.color.b = 0.0;}
          else if (class_id == 5){text.text = "B2"; text.color.r = 1.0; text.color.g = 0.0; text.color.b = 1.0;}
          else if (class_id == 6){text.text = "B3"; text.color.r = 0.0; text.color.g = 1.0; text.color.b = 1.0;}
          else{text.text = "None";}
          text.id = i;
          text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
          text.color.a = 1.0;
          text.scale.z = 1.0;
          text.action = visualization_msgs::msg::Marker::ADD;
          text.pose.orientation.w = 1.0;
          text.pose.position.x = coord_vector[0].x;
          text.pose.position.y = coord_vector[0].y;
          text.lifetime.nanosec = 100000000;
          text_array.markers.push_back(text);
        }
        else if (coord_vector.size() > 1)
        {
          cout << "클러스터 두 개 잡힘!!" << endl;
          float min_dist = 100.0f;
          int min_index = -1;
          for (int k = 0; k < coord_vector.size(); k++)
          {
            float x = coord_vector[k].x;
            float y = coord_vector[k].y;
            float dist_ = sqrt(pow(x, 2) + pow(y, 2));

            if (dist_ < min_dist)
            {
              min_index = k;
              min_dist = dist_;
            }
          }
          // float data
          coord.data.push_back(coord_vector[min_index].x);
          coord.data.push_back(coord_vector[min_index].y);
          coord.data.push_back(coord_vector[min_index].id);
          // pcl data
          pcl::PointXYZ point;
          point.x = coord_vector[min_index].x;
          point.y = coord_vector[min_index].y;
          point.z = 0.0f;
          coord_cloud->push_back(point);
          // marker data
          visualization_msgs::msg::Marker text;
          text.header.frame_id = "velodyne";
          text.ns = "text";
          if (class_id == 1){text.text = "A1"; text.color.r = 1.0; text.color.g = 0.0; text.color.b = 0.0;}
          else if (class_id == 2){text.text = "A2"; text.color.r = 0.0; text.color.g = 1.0; text.color.b = 0.0;}
          else if (class_id == 3){text.text = "A3"; text.color.r = 0.0; text.color.g = 0.0; text.color.b = 1.0;}
          else if (class_id == 4){text.text = "B1"; text.color.r = 1.0; text.color.g = 1.0; text.color.b = 0.0;}
          else if (class_id == 5){text.text = "B2"; text.color.r = 1.0; text.color.g = 0.0; text.color.b = 1.0;}
          else if (class_id == 6){text.text = "B3"; text.color.r = 0.0; text.color.g = 1.0; text.color.b = 1.0;}
          else{text.text = "None";}
          text.id = i;
          text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
          text.color.a = 1.0;
          text.scale.z = 1.0;
          text.action = visualization_msgs::msg::Marker::ADD;
          text.pose.orientation.w = 1.0;
          text.pose.position.x = coord_vector[min_index].x;
          text.pose.position.y = coord_vector[min_index].y;
          text.lifetime.nanosec = 100000000;
          text_array.markers.push_back(text);
        }
      }

      marker_pub_->publish(text_array);
      text_array.markers.clear();

      // 플래닝으로 가는 데이터
      this->publish_cone_->publish(coord);

      // 포인트클라우드 퍼블리시
      coord_cloud->width = 1;
      coord_cloud->height = coord_cloud->points.size();
      pcl::toROSMsg(*coord_cloud, cloud_msg);
      cloud_msg.header.frame_id = "velodyne";
      // cloud_msg.stamp = node->now();
      this->publish_point_->publish(cloud_msg);

      //====== 클러스터 박스 초기화 ==========
      this->box_locker = 0;
      boxes.clear();
      //==================================
      //======== 욜로박스 초기화 ============
      this->yolo_locker = 0;
      boxes_yolo.clear();
      //==================================
    }
    this->sync_locker = 0;
  } 
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageLiDARFusion>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}