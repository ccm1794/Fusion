#!/usr/bin/env/ python3

import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
#from launch_ros.actions import DeclareLaunchArgument
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    param_dir = LaunchConfiguration(
        'param_dir',
        default=os.path.join(
        get_package_share_directory('fusion'),
        'param',
        'B_amz.yaml'))
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'param_dir',
            default_value=param_dir,
            description='Full path to param file to load'),

        Node(
            package='fusion',
            executable='lidar_pre_amz_sync',
            name='lidar_pre_amz_sync',
            output='screen'),

        Node(
            package='fusion',
            executable='amz_fusion_sync',
            name='amz_fusion_sync',
            parameters=[param_dir],
            output='screen'),

        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            namespace='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d' + os.path.join(
                get_package_share_directory('fusion'),
                'rviz',
                'two_bag.rviz'
            )]
        )
    ])        
        
    
