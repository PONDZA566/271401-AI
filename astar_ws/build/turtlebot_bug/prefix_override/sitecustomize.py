import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pondza/271401-AI/astar_ws/install/turtlebot_bug'
