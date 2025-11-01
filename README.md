# 271401-AI
## A* path planning + DETOUR Simulation Guide

## 1.Spawn Turtlebot
```
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

## 2.Generate Real-time Mapping
```
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```
## 3.Run Node
```
ros2 run astar_planner a_star_planner
```