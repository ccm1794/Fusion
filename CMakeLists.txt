cmake_minimum_required(VERSION 3.5)
project(fusion)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)

set(dependencies
  rclcpp
  std_msgs
  PCL
  pcl_conversions
  cv_bridge
  geometry_msgs
  visualization_msgs
  sensor_msgs
  vision_msgs
  OpenCV
  message_filters
)

link_libraries(
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${pcl_ros_LIBRARIES}
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

#etc
add_executable(image_publisher src/etc/image_publisher.cpp)
ament_target_dependencies(image_publisher ${dependencies})

add_executable(projection_video src/etc/projection_video.cpp)
ament_target_dependencies(projection_video ${dependencies})

add_executable(projection_flir src/etc/projection_flir.cpp)
ament_target_dependencies(projection_flir ${dependencies})

#deli
add_executable(deli_fusion src/deli/deli_fusion.cpp)
ament_target_dependencies(deli_fusion ${dependencies})

add_executable(deli_fusion_beta src/deli/deli_fusion_beta.cpp)
ament_target_dependencies(deli_fusion_beta ${dependencies})

add_executable(deli_fusion_gamma src/deli/deli_fusion_gamma.cpp)
ament_target_dependencies(deli_fusion_gamma ${dependencies})

add_executable(deli_fusion_sync src/deli/deli_fusion_sync.cpp)
ament_target_dependencies(deli_fusion_sync ${dependencies})

add_executable(compare_box_image_deliver src/deli/compare_box_image_deliver.cpp)
ament_target_dependencies(compare_box_image_deliver ${dependencies})

add_executable(compare_box_image_deliver_beta src/deli/compare_box_image_deliver_beta.cpp)
ament_target_dependencies(compare_box_image_deliver_beta ${dependencies})

add_executable(compare_box_image_deliver_logi src/deli/compare_box_image_deliver_logi.cpp)
ament_target_dependencies(compare_box_image_deliver_logi ${dependencies})

#amz
add_executable(amz_fusion src/amz/amz_fusion.cpp)
ament_target_dependencies(amz_fusion ${dependencies})

add_executable(amz_fusion_beta src/amz/amz_fusion_beta.cpp)
ament_target_dependencies(amz_fusion_beta ${dependencies})

add_executable(amz_fusion_gamma src/amz/amz_fusion_gamma.cpp)
ament_target_dependencies(amz_fusion_gamma ${dependencies})

add_executable(amz_fusion_image_gamma src/amz/amz_fusion_image_gamma.cpp)
ament_target_dependencies(amz_fusion_image_gamma ${dependencies})

add_executable(amz_fusion_sync src/amz/amz_fusion_sync.cpp)
ament_target_dependencies(amz_fusion_sync ${dependencies})

add_executable(compare_box_image_amz_beta src/amz/compare_box_image_amz_beta.cpp)
ament_target_dependencies(compare_box_image_amz_beta ${dependencies})

add_executable(compare_box_image_amz src/amz/compare_box_image_amz.cpp)
ament_target_dependencies(compare_box_image_amz ${dependencies})

add_executable(compare_box_image_amz_sync src/amz/compare_box_image_amz_sync.cpp)
ament_target_dependencies(compare_box_image_amz_sync ${dependencies})

add_executable(black src/amz/black.cpp)
ament_target_dependencies(black ${dependencies})

add_executable(black_rightup src/amz/black_rightup.cpp)
ament_target_dependencies(black_rightup ${dependencies})

#lidar_pre
add_executable(lidar_pre_amz src/lidar_pre/lidar_pre_amz.cpp)
ament_target_dependencies(lidar_pre_amz ${dependencies})

add_executable(lidar_pre_amz_sync src/lidar_pre/lidar_pre_amz_sync.cpp)
ament_target_dependencies(lidar_pre_amz_sync ${dependencies})

add_executable(lidar_pre_deli src/lidar_pre/lidar_pre_deli.cpp)
ament_target_dependencies(lidar_pre_deli ${dependencies})

install(TARGETS

  image_publisher
  projection_flir
  projection_video
  
  deli_fusion
  deli_fusion_beta
  deli_fusion_gamma
  deli_fusion_sync
  compare_box_image_deliver
  compare_box_image_deliver_beta
  compare_box_image_deliver_logi
  
  amz_fusion
  amz_fusion_gamma
  amz_fusion_image_gamma
  amz_fusion_beta
  amz_fusion_sync
  compare_box_image_amz
  compare_box_image_amz_beta
  compare_box_image_amz_sync
  black
  black_rightup

  lidar_pre_amz
  lidar_pre_amz_sync
  lidar_pre_deli

  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  param
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
