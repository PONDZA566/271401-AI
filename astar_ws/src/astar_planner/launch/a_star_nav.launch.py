from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='astar_planner',
            executable='a_star_planner',
            output='screen'
        ),
    ])
