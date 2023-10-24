#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat frame1, frame2, image1, image2;
void on_mouse(int event, int x, int y, int flags, void *);

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("image_publisher");

  auto publisher1 = node->create_publisher<sensor_msgs::msg::Image>("video1", 10);
  sensor_msgs::msg::Image::SharedPtr msg1;

  auto publisher2 = node->create_publisher<sensor_msgs::msg::Image>("video2", 10);
  sensor_msgs::msg::Image::SharedPtr msg2;

  auto pub = node->create_publisher<sensor_msgs::msg::Image>("correct_image", 10);

  VideoCapture cap1("/dev/v4l/by-id/usb-046d_罗技高清网络摄像机_C930c_744E4AAE-video-index0",CAP_V4L);
  cap1.set(CAP_PROP_FRAME_WIDTH, 640);
  cap1.set(CAP_PROP_FRAME_HEIGHT, 480);

  VideoCapture cap2("/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_964D8E9E-video-index0",CAP_V4L);
  cap2.set(CAP_PROP_FRAME_WIDTH, 640);
  cap2.set(CAP_PROP_FRAME_HEIGHT, 480);

  Mat merged;

  rclcpp::WallRate loop_rate(20.0);

  while(rclcpp::ok())
  {
    cap1 >> frame1;
    cap2 >> frame2;

    msg1 = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame1).toImageMsg();
    publisher1->publish(*msg1.get()); 
    msg2 = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame2).toImageMsg();
    publisher2->publish(*msg2.get()); 

    image1 = cv_bridge::toCvShare(msg1)->image;

    image2 = cv_bridge::toCvShare(msg2)->image;

    vconcat(image1, image2, merged);
    sensor_msgs::msg::Image::UniquePtr image_msg = std::make_unique<sensor_msgs::msg::Image>();
    image_msg->height = merged.rows;
    image_msg->width = merged.cols;
    image_msg->encoding = "bgr8";
    image_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(merged.step);
    size_t size = merged.step * merged.rows;
    image_msg->data.resize(size);
    memcpy(&image_msg->data[0], merged.data, size);

    pub->publish(std::move(image_msg));

    rclcpp::spin_some(node);
    loop_rate.sleep();

    imshow("merged", merged);
    namedWindow("merged");

    cv::waitKey(10);

    char ch = cv::waitKey(10);
    if(ch == 27) break;
  }

}