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

#include <sensor_msgs/msg/point_cloud.hpp>8
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
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <vision_msgs/msg/bounding_box3_d.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

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

// 클러스터 박스 처리를 위한 구조체
struct Box_points
{
  float x1,y1,z1;
  float x2,y2,z2;
  float x3,y3,z3;
  float x4,y4,z4;
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

// 클러스터 박스를 크기순대로 정렬하기 위한 함수
bool compareBoxes(const Box& a, const Box& b) 
{
  return a.x < b.x;
}

// 클러스터 박스의 8개의 꼭지점을 구하는 함수
Box_points calcBox_points(const Box &box)
{
  Box_points vertices;
  if(box.y >= 0)
  {
    vertices.x1 = box.x - (box.size_x)/2;
    vertices.y1 = box.y + (box.size_y)/2;
    vertices.z1 = box.z - (box.size_z)/2;

    vertices.x2 = box.x + (box.size_x)/2;
    vertices.y2 = box.y - (box.size_y)/2;
    vertices.z2 = box.z - (box.size_z)/2;

    vertices.x3 = box.x + (box.size_x)/2;
    vertices.y3 = box.y - (box.size_y)/2;
    vertices.z3 = box.z + (box.size_z)/2;

    vertices.x4 = box.x - (box.size_x)/2;
    vertices.y4 = box.y + (box.size_y)/2;
    vertices.z4 = box.z + (box.size_z)/2;
  }
  else if(box.y < 0)
  {
    vertices.x1 = box.x + (box.size_x)/2;
    vertices.y1 = box.y + (box.size_y)/2;
    vertices.z1 = box.z - (box.size_z)/2;

    vertices.x2 = box.x - (box.size_x)/2;
    vertices.y2 = box.y - (box.size_y)/2;
    vertices.z2 = box.z - (box.size_z)/2;

    vertices.x3 = box.x - (box.size_x)/2;
    vertices.y3 = box.y - (box.size_y)/2;
    vertices.z3 = box.z + (box.size_z)/2;

    vertices.x4 = box.x + (box.size_x)/2;
    vertices.y4 = box.y + (box.size_y)/2;
    vertices.z4 = box.z + (box.size_z)/2;
  }
  return vertices;
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

std::mutex mut_img, mut_box, mut_yolo;
std::vector<Box> boxes;
std::vector<Box_yolo> boxes_yolo;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  Mat transformMat;
  Mat CameraMat;
  Mat DistCoeff;

  Mat frame;
  Mat image_undistorted;

  Mat overlay; // 전체 오버레이

  bool is_rec_image = false;
  bool is_rec_box = false;
  bool is_rec_yolo = false;

  int obj_count;
  int cluster_count;

  int box_locker = 0; // box_Callback 할 때 lock해주는 변수
  int yolo_locker = 0;

  float min_iou = 0.2;
  float min_area = 100;
  
  vector<double> CameraExtrinsic_vector;
  vector<double> CameraMat_vector;
  vector<double> DistCoeff_vector;

public:
  ImageLiDARFusion()
  : Node("projection_box")
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

    // 원본이미지 subscribe
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/video1", 1,
      [this](const sensor_msgs::msg::Image::SharedPtr msg) -> void
      {
        ImageCallback(msg);
      });

    // 실질적인 yolo 박스 subscribe
    yolo_detect_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
      "/yolo_detect", 1,
      [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) -> void
      {
        YOLOCallback(msg);
      });
    
    // 실질적으로 클러스터 박스 subscribe
    lidar_box_sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
      "/lidar_bbox", 1,
      [this](const vision_msgs::msg::Detection3DArray::SharedPtr msg) -> void
      {
        BoxCallback(msg);
      });

    // 결과 퍼블리시
    publish_cone_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/coord_xy", 1); // origin qos = 10

    publish_point_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/coord_xyz", 1); // origin qos = 10

    // 타이머 콜백으로 박스 매칭함수 실행
    auto timer_callback = [this]() -> void {FusionCallback();};
    timer_ = create_wall_timer(80ms, timer_callback); // 10hz
    
    RCLCPP_INFO(this->get_logger(), "------------initialize end------------\n");
  }

  ~ImageLiDARFusion()
  {
    // pthread_join(this->tids1_, NULL);
    // pthread_join(this->tids2_, NULL);
  }

public:
  void set_param();
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void YOLOCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);
  void BoxCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
  void FusionCallback();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yolo_detect_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr lidar_box_sub_;  
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publish_cone_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publish_point_;
  rclcpp::TimerBase::SharedPtr timer_;
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

// image callback
void ImageLiDARFusion::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  Mat image;

  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
  }
  catch(cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  mut_img.lock();
  cv::undistort(cv_ptr->image, this->image_undistorted, this->CameraMat, this->DistCoeff);
  // this->overlay = this->image_undistorted.clone(); // 이거 무슨뜻?
  mut_img.unlock();

  this->is_rec_image = true;

}

// yolo 박스 설정
void ImageLiDARFusion::YOLOCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  if (msg->detections.size() != 0 && this->yolo_locker == 0)
  {
    this->yolo_locker = 1;
    std::string Class_name;
    // cout << "yolo count : " << this->obj_count << endl;
    // cout << "sizeof() : " << msg->detections.size() << endl;
    

    for (int i = 0; i <msg->detections.size(); i++)
    {
      Class_name = msg->detections[i].results[0].id; // 클래스 가져오기

      int color = 0;

      if(Class_name == "blue")
      {
        color = 1;
      }
      else if(Class_name == "orange")
      {
        color = 2;
      }
      else
      {
        color = 0;
      }

      float area = msg->detections[i].bbox.size_x * msg->detections[i].bbox.size_y;
      if(area > this->min_area)
      {      
        mut_yolo.lock();
        Box_yolo box_yolo = 
        {
          msg->detections[i].bbox.center.x - (msg->detections[i].bbox.size_x)/2 * 1.2, // x1
          msg->detections[i].bbox.center.x + (msg->detections[i].bbox.size_x)/2 * 1.2, // x2
          msg->detections[i].bbox.center.y - (msg->detections[i].bbox.size_y)/2, // y1
          msg->detections[i].bbox.center.y + (msg->detections[i].bbox.size_y)/2, // y2
          color
        };
        boxes_yolo.push_back(box_yolo);
        mut_yolo.unlock();
      }
    }
    this->is_rec_yolo = true;
    
  }
  else{}
}

// Box callback
void ImageLiDARFusion::BoxCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg) // 사용자 정의 인터페이스 만들 필요 없다
{
  this->cluster_count = msg->detections.size();

  // cout << "cluster count : " << cluster_count << endl;
  // 메세지가 얼마나 빠르게 들어오나 확인용 : velodyne_cluster.cpp에서 쏠 때 시간과 큰 차이가 없음.(0.02초)
  // RCLCPP_INFO(this->get_logger(), "cluster size : %d", cluster_count);  

  if(this->cluster_count != 0 && this->box_locker == 0)
  {
    for(int i = 0; i < this->cluster_count; i++)
    { 
      // if( msg->detections[i].bbox.size.z / msg->detections[i].bbox.size.y < 4 && msg->detections[i].bbox.size.y < 0.45 && msg->detections[i].bbox.size.z < 0.8)
      if(msg->detections[i].bbox.size.z < 1.5 && msg->detections[i].bbox.size.y < 0.5)
      {
        mut_box.lock();
        Box box = 
        {
          msg->detections[i].bbox.center.position.x, 
          msg->detections[i].bbox.center.position.y, 
          msg->detections[i].bbox.center.position.z, 
          msg->detections[i].bbox.size.x, 
          msg->detections[i].bbox.size.y,  // 투영되는 박스 크기를 임의로 변형
          msg->detections[i].bbox.size.z // 투영되는 박스 크기를 임의로 변형
        };
        boxes.push_back(box);
        mut_box.unlock();    
      }
    }
    this->box_locker = 1;
    this->is_rec_box = true;
  }
}

// fusion callback
void ImageLiDARFusion::FusionCallback()
{
  mut_img.lock();
  this->overlay = this->image_undistorted.clone();
  Mat frame_projected = this->image_undistorted.clone();
  mut_img.unlock();

  mut_box.lock();
  std::vector<Box> lidar_boxes = boxes;
  mut_box.unlock();

  mut_yolo.lock();
  std::vector<Box_yolo> yolo_boxes = boxes_yolo;
  mut_yolo.unlock();

  if (this->yolo_locker == 1 && this->box_locker == 1)
  {
    std::vector<Box_yolo> boxes_2d_cluster;

    for (const auto& Box : lidar_boxes)
    {
      Box_points vertices = calcBox_points(Box);
      double box_1[4] = {vertices.x1, vertices.y1, vertices.z1, 1.0};
      double box_2[4] = {vertices.x2, vertices.y2, vertices.z2, 1.0};
      double box_3[4] = {vertices.x3, vertices.y3, vertices.z3, 1.0};
      double box_4[4] = {vertices.x4, vertices.y4, vertices.z4, 1.0};

      cv::Mat pos1( 4, 1, CV_64F, box_1); // 3차원 좌표
      cv::Mat pos2( 4, 1, CV_64F, box_2);
      cv::Mat pos3( 4, 1, CV_64F, box_3);
      cv::Mat pos4( 4, 1, CV_64F, box_4);

      //카메라 원점 xyz 좌표 (3,1)생성
      cv::Mat newpos1(this->transformMat * pos1); // 카메라 좌표로 변환한 것.
      cv::Mat newpos2(this->transformMat * pos2);
      cv::Mat newpos3(this->transformMat * pos3);
      cv::Mat newpos4(this->transformMat * pos4);

      float x1 = (float)(newpos1.at<double>(0, 0) / newpos1.at<double>(2, 0));
      float y1 = (float)(newpos1.at<double>(1, 0) / newpos1.at<double>(2, 0));

      float x2 = (float)(newpos2.at<double>(0, 0) / newpos2.at<double>(2, 0));
      float y2 = (float)(newpos2.at<double>(1, 0) / newpos2.at<double>(2, 0));

      float x3 = (float)(newpos3.at<double>(0, 0) / newpos3.at<double>(2, 0));
      float y3 = (float)(newpos3.at<double>(1, 0) / newpos3.at<double>(2, 0));

      float x4 = (float)(newpos4.at<double>(0, 0) / newpos4.at<double>(2, 0));
      float y4 = (float)(newpos4.at<double>(1, 0) / newpos4.at<double>(2, 0));

      if (Box.y >= 0)
      {
        // cv::rectangle(this->overlay, Rect(Point(x4, y3), Point(x2, y1)), Scalar(0, 255, 0), 2, 8, 0);
        // Box_yolo box_basic = { x4, x2, y3, y1 };
        // boxes_2d_cluster.push_back(box_basic);

        cv::Point pt1;
        cv::Point pt2;
        cv::Point pt3;
        cv::Point pt4;

        pt1.x = x1;
        pt1.y = y1;
        cv::circle(this->overlay, pt1, 6, cv::Scalar(0, 0, 255), -1);

        pt2.x = x2;
        pt2.y = y2;
        cv::circle(this->overlay, pt2, 6, cv::Scalar(255, 0, 0), -1);

        pt3.x = x3;
        pt3.y = y3;
        cv::circle(this->overlay, pt3, 6, cv::Scalar(0, 255, 0), -1);

        pt4.x = x4;
        pt4.y = y4;
        cv::circle(this->overlay, pt4, 6, cv::Scalar(0, 255, 255), -1);
        Box_yolo box_basic = { x4, x2, y3, y1 };
        boxes_2d_cluster.push_back(box_basic);
      }
      else if (Box.y < 0)
      {
        // cv::rectangle(this->overlay, Rect(Point(x4, y4), Point(x2, y2)), Scalar(0, 255, 0), 2, 8, 0);
        // Box_yolo box_basic = { x4, x2, y4, y2 };
        // boxes_2d_cluster.push_back(box_basic);
        cv::Point pt1;
        cv::Point pt2;
        cv::Point pt3;
        cv::Point pt4;

        pt1.x = x1;
        pt1.y = y1;
        cv::circle(this->overlay, pt1, 6, cv::Scalar(0, 0, 255), -1);

        pt2.x = x2;
        pt2.y = y2;
        cv::circle(this->overlay, pt2, 6, cv::Scalar(255, 0, 0), -1);

        pt3.x = x3;
        pt3.y = y3;
        cv::circle(this->overlay, pt3, 6, cv::Scalar(0, 255, 0), -1);

        pt4.x = x4;
        pt4.y = y4;
        cv::circle(this->overlay, pt4, 6, cv::Scalar(0, 255, 255), -1);

        Box_yolo box_basic = { x4, x2, y4, y2 };
        boxes_2d_cluster.push_back(box_basic);
      }      
    }

    for (const auto& Box : yolo_boxes)
    {
      int xx1 = Box.x1;
      int xx2 = Box.x2;
      int yy1 = Box.y1;
      int yy2 = Box.y2;

      if (Box.color == 1)
      {
        cv::rectangle(this->overlay, Rect(Point(xx1, yy1), Point(xx2, yy2)), Scalar(0, 0, 255), 1, 8, 0);
      }
      else if (Box.color == 2)
      {
        cv::rectangle(this->overlay, Rect(Point(xx1, yy1), Point(xx2, yy2)), Scalar(255, 0, 0), 1, 8, 0);
      }
    }

    // 이미지에 투영하는 부분
    float opacity = 0.6;
    cv::addWeighted(this->overlay, opacity, frame_projected, 1-opacity, 0, frame_projected); // 투영된 박스를 합친다.
  
    std_msgs::msg::Float32MultiArray coord;
    sensor_msgs::msg::PointCloud2 cloud_msg;

    pcl::PointCloud<pcl::PointXYZI>::Ptr coord_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    for ( int i = 0; i < boxes_2d_cluster.size(); i++)
    {
      float max_iou = 0.0f;
      int class_id = -1;

      for (int j = 0; j < yolo_boxes.size(); j++)
      {
        float iou = get_iou(boxes_2d_cluster[i], yolo_boxes[j]);
        if(iou > max_iou)
        {
          max_iou = iou;
          class_id = yolo_boxes[j].color;
        }
      }

      if (max_iou > this->min_iou)
      {
        pcl::PointXYZI coord_;

        cv::rectangle(frame_projected, 
        Rect(Point(boxes_2d_cluster[i].x1, boxes_2d_cluster[i].y1), 
        Point(boxes_2d_cluster[i].x2, boxes_2d_cluster[i].y2)), 
        Scalar(100, 100, 0), 3, 8, 0);

        float center_x = lidar_boxes[i].x; // 3차원 상대좌표에서 x,y 가져옴
        float center_y = lidar_boxes[i].y;

        // float data
        coord.data.push_back(center_x); 
        coord.data.push_back(center_y);
        coord.data.push_back(class_id);
        coord.data.push_back(-1000.0);

        // pcl data
        coord_.x = center_x;
        coord_.y = center_y;
        coord_.z = 0.;
        if (class_id == 1)
        {
          coord_.intensity = 0.5;
        }
        else if (class_id == 2)
        {
          coord_.intensity = 0.7;
        }

        coord_cloud->push_back(coord_);
      }
    }
    //imshow
    string windowName = "overlay";
    cv::namedWindow(windowName, 3);
    cv::imshow(windowName, frame_projected);
    char ch = cv::waitKey(1);
    // if(ch == 27) break;

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

    this->is_rec_image = false;
    this->is_rec_box = false;
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