// https://github.com/williamhyin/lidar_to_camera/blob/master/src/project_lidar_to_camera.cpp 투영은 이거 참고함

#include <rclcpp/rclcpp.hpp>
#include <mutex>
#include <memory>
#include <thread>
#include <pthread.h>

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
#include <vision_msgs/msg/detection2_d_array.hpp>

using namespace std;
using namespace cv;

static bool IS_IMAGE_CORRECTION = true;

std::mutex mut_img;
std::mutex mut_pc;

pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr copy_raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);

// 카메라 이미지
cv::Mat image_color;
cv::Mat overlay;
cv::Mat copy_image_color;
cv::Mat img_HSV;

// Yolo Global parameter
std::mutex mut_yolo;
std::string Class_name;

int obj_count;

bool is_rec_LiDAR = false;
int lidar_button;

struct Box_yolo
{
  float x1;
  float x2;
  float y1;
  float y2;
  int color;
};
std::vector<Box_yolo> boxes_yolo;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  vector<double> CameraExtrinsic_vector;
  vector<double> CameraMat_vector;
  vector<double> DistCoeff_vector;
  vector<int> ImageSize_vector;

  Mat concatedMat;
  Mat transformMat;
  Mat CameraExtrinsicMat;
  Mat CameraMat;
  Mat DistCoeff;

  pthread_t tids1_; //스레드 아이디

  int img_height = 480; // 왜 int는 못불러 오는가...
  int img_width = 640;

  float maxlen =200.0;         /**< Max distance: LiDAR */
  float minlen = 0.0001;        /**< Min distance: LiDAR */
  float max_FOV = CV_PI/2;     /**< Max FOV : Camera */

  sensor_msgs::msg::PointCloud2 colored_msg;

public:
  ImageLiDARFusion()
  : Node("projection")
  {
    RCLCPP_INFO(this->get_logger(), "------------ intialize ------------\n");

    this->declare_parameter("CameraExtrinsicMat", vector<double>());
    this->CameraExtrinsic_vector = this->get_parameter("CameraExtrinsicMat").as_double_array();
    this->declare_parameter("CameraMat", vector<double>());
    this->CameraMat_vector = this->get_parameter("CameraMat").as_double_array();
    this->declare_parameter("DistCoeff", vector<double>());
    this->DistCoeff_vector = this->get_parameter("DistCoeff").as_double_array();


    this->set_param();

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
       "/flir_camera/image_raw", 10,
       [this](const sensor_msgs::msg::Image::SharedPtr msg) -> void
       {
         ImageCallback(msg);
       }); //람다함수를 사용했는데 왜?, 그리고 일반적인 방법으로 하면 동작하지 않는다. 이유를 모르겠다

    LiDAR_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/velodyne_points", 10,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) -> void
      {
        LiDARCallback(msg);
      });

    // int ret1 = pthread_create(&this->tids1_, NULL, publish_thread, this);
    RCLCPP_INFO(this->get_logger(), "------------ intialize end------------\n");
  }

  ~ImageLiDARFusion()
  {
    // pthread_join(this->tids1_, NULL);
  }

public:
  void set_param();
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_sub_;
};


void ImageLiDARFusion::set_param()
{
  // 받아온 파라미터
  //Mat CameraExtrinsicMat;
  // Mat concatedMat;

  Mat CameraExtrinsicMat_(4,4, CV_64F, CameraExtrinsic_vector.data());
  Mat CameraMat_(3,3, CV_64F, CameraMat_vector.data());
  Mat DistCoeffMat_(1,4, CV_64F, DistCoeff_vector.data());

  //위에 있는 내용 복사
  CameraExtrinsicMat_.copyTo(this->CameraExtrinsicMat);
  CameraMat_.copyTo(this->CameraMat);
  DistCoeffMat_.copyTo(this->DistCoeff);

  //재가공 : 회전변환행렬, 평행이동행렬
  Mat Rlc = this->CameraExtrinsicMat(cv::Rect(0,0,3,3));
  Mat Tlc = this->CameraExtrinsicMat(cv::Rect(3,0,1,3));

  cv::hconcat(Rlc, Tlc, this->concatedMat);

  this->transformMat = this->CameraMat * this->concatedMat;

  RCLCPP_INFO(this->get_logger(), "transform Matrix ready_");
}

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
  cv_ptr->image.copyTo(image);
  cv::resize(image, image, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
  cv::undistort(image, image_color, this->CameraMat, this->DistCoeff);
  image_color = image_color.clone();
  overlay = image_color.clone();
  mut_img.unlock();

  mut_pc.lock();
  pcl::copyPointCloud(*raw_pcl_ptr, *copy_raw_pcl_ptr);
  mut_pc.unlock();

  const int size = copy_raw_pcl_ptr->points.size();

  for (int i = 0; i < size ; i++)
  {
    pcl::PointXYZ temp_;
    temp_.x = copy_raw_pcl_ptr->points[i].x;
    temp_.y = copy_raw_pcl_ptr->points[i].y;
    temp_.z = copy_raw_pcl_ptr->points[i].z;

    float azimuth_ = abs(atan2(temp_.y, temp_.x)); //atan2(y,x) : y가 -면 -값 나옴 +면 +값 나옴

    if(azimuth_ > max_FOV)
    {
      continue; //처리하지 않겠다
    }
    else if(azimuth_ <= max_FOV)
    {
      double a_[4] = {temp_.x, temp_.y, temp_.z, 1.0};
      cv::Mat pos( 4, 1, CV_64F, a_); // 라이다 좌표
      cv::Mat newpos(transformMat * pos); // 카메라 좌표로 변환한 것.

      float x = (float)(newpos.at<double>(0, 0) / newpos.at<double>(2, 0));
      float y = (float)(newpos.at<double>(1, 0) / newpos.at<double>(2, 0));

      float dist_ = sqrt(pow(temp_.x,2) + pow(temp_.y,2) + pow(temp_.z,2));

      if (minlen < dist_ && dist_ < maxlen)
      {
        if (x >= 0 && x < img_width && y >= 0 && y < img_height)
        {
          int row = int(y);
          int column = int(x);

          cv::Point pt;
          pt.x = x;
          pt.y = y;

          float val = temp_.x; // 라이다 좌표에서 x를 뜻함
          float maxVal = 100.0;

          int green = min(255, (int) (255 * abs((val - maxVal) / maxVal)));
          int red = min(255, (int) (255 * (1 - abs((val - maxVal) / maxVal))));
          cv::circle(overlay, pt, 2, cv::Scalar(0, green, red), -1);
        }
        else{}
      } 
    } 
  }
  float opacity = 0.6;
  cv::addWeighted(overlay, opacity, image_color, 1 - opacity, 0, image_color);

  string windowName = "overlay";
  cv::namedWindow(windowName, 3);
  cv::imshow(windowName, image_color); 
  char ch = cv::waitKey(10);
  lidar_button = 0;
}
void ImageLiDARFusion::LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (lidar_button == 0)
  {
    lidar_button = 1;
    mut_pc.lock();
    pcl::fromROSMsg(*msg, *raw_pcl_ptr);
    mut_pc.unlock();
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