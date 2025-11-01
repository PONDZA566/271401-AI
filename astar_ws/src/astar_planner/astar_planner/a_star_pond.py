#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np
import heapq, math, time

# ✅ QoS imports
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    qos_profile_sensor_data,
)

def rotate2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

class AStarTurtleBot(Node):
    def __init__(self):
        super().__init__('a_star_turtlebot')

        # === ROS Setup ===
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ✅ รับแผนที่ latched
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)

        # ✅ รับ Lidar (sensor QoS)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)

        # ✅ TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.scan_frame = 'base_link'  # เปลี่ยนเป็น 'base_scan' ถ้า Lidar อยู่เฟรมนี้

        # === Map Info ===
        self.map = None
        self.res = None
        self.origin = None
        self.width = None
        self.height = None
        self.map_ready = False

        # === Dynamic obstacles mask ===
        self.dynamic_mask = None          # np.bool_ [H, W]
        self.inflated_mask = None         # np.bool_ [H, W]
        self.inflation_radius_m = 0.25    # เมตร (บัฟเฟอร์ความปลอดภัยรอบสิ่งกีดขวาง)

        # === Robot State ===
        self.pose = np.array([0.0, 0.0, 0.0])
        self.path = []
        self.path_idx = 0
        self.path_ready = False

        # === Parameters ===
        self.goal_distance = 3.5         # ระยะเป้าหมายข้างหน้า (เมตร)
        self.linear_speed = 0.18
        self.angular_speed = 1.2
        self.dist_tol = 0.15
        self.angle_tol = 0.25

        # === Laser Info / avoidance ===
        self.front_min_dist = float('inf')
        self.obstacle_threshold = 0.35    # ถ้าสิ่งกีดขวางอยู่ใกล้กว่านี้ → replan
        self.emergency_stop_dist = 0.22   # หยุดฉุกเฉิน
        self.last_obstacle_time = 0.0
        self.replan_cooldown = 2.0
        self.is_replanning = False

        # ⭐️ LiDAR usable range & front arc (ปรับได้ตอนรัน)
        self.declare_parameter('lidar_use_min', 0.12)   # m (ตั้งตามรุ่น/คุณภาพจริง)
        self.declare_parameter('lidar_use_max', 0.60)   # m (ใช้ข้อมูลแค่ภายในระยะนี้)
        self.declare_parameter('front_arc_deg', 120.0)  # ใช้มุมด้านหน้ากี่องศา (สำหรับ near/mark)
        self.lidar_use_min = float(self.get_parameter('lidar_use_min').value)
        self.lidar_use_max = float(self.get_parameter('lidar_use_max').value)
        self.front_arc_deg = float(self.get_parameter('front_arc_deg').value)

        # เก็บพารามิเตอร์จากสแกนล่าสุด (พร้อมกรองระยะแล้ว)
        self.scan_last = None   # (angle_min, angle_inc, ranges_filtered, rmin_eff, rmax_eff)

        # Timer main loop
        self.create_timer(0.1, self.loop)
        self.get_logger().info("🟢 A* + Dynamic Obstacle Avoidance started")

    # -------------------- Map Callback --------------------
    def on_map(self, msg):
        self.map = np.array(msg.data, dtype=int).reshape(msg.info.height, msg.info.width)
        self.res = msg.info.resolution
        self.origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.width = msg.info.width
        self.height = msg.info.height
        self.map_ready = True

        # เตรียมมาสก์ไดนามิกให้ขนาดตรงกับแผนที่
        self.dynamic_mask = np.zeros((self.height, self.width), dtype=bool)
        self.inflated_mask = np.zeros_like(self.dynamic_mask)

    # -------------------- Laser Callback --------------------
    def on_scan(self, msg: LaserScan):
        # กำหนดช่วงระยะที่ "ยอมรับ" เพื่อนำมาใช้จริง
        rmin_hw = msg.range_min if msg.range_min > 0 else 0.0
        rmax_hw = msg.range_max if msg.range_max > 0 else float('inf')
        rmin_eff = max(rmin_hw, self.lidar_use_min)
        rmax_eff = min(rmax_hw, self.lidar_use_max)

        # หน้าต่างตรงกลางสำหรับ "front_min_dist" (เฉพาะรังสีในช่วงระยะที่ยอมรับ)
        center = len(msg.ranges) // 2
        left = max(0, center - 20)
        right = min(len(msg.ranges), center + 20)
        window = msg.ranges[left:right]

        valid = []
        for r in window:
            if r is None:
                continue
            if math.isfinite(r) and (rmin_eff <= r <= rmax_eff):
                valid.append(r)
        self.front_min_dist = float(np.median(valid)) if valid else float('inf')

        # เก็บข้อมูลสแกน "หลังกรองระยะแล้ว" เพื่อใช้สร้าง dynamic mask
        ranges_filtered = []
        for r in msg.ranges:
            if r is None or (not math.isfinite(r)) or (r < rmin_eff) or (r > rmax_eff):
                ranges_filtered.append(None)   # ข้ามลำแสงนี้
            else:
                ranges_filtered.append(r)      # เก็บค่าระยะที่อยู่ในช่วงเท่านั้น

        self.scan_last = (msg.angle_min, msg.angle_increment, ranges_filtered, rmin_eff, rmax_eff)

    # -------------------- TF lookup --------------------
    def update_pose(self):
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.pose = np.array([x, y, yaw])
            return True
        except Exception:
            return False

    # -------------------- Grid Conversion --------------------
    def world_to_grid(self, pos):
        gx = int((pos[0] - self.origin[0]) / self.res)
        gy = int((pos[1] - self.origin[1]) / self.res)
        return (gx, gy)

    def grid_to_world(self, grid):
        wx = grid[0] * self.res + self.origin[0]
        wy = grid[1] * self.res + self.origin[1]
        return np.array([wx, wy])

    # -------------------- Obstacle Check --------------------
    def is_free(self, grid):
        x, y = grid
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        v = self.map[y, x]
        # อนุญาต unknown (-1) แต่ห้ามชนสิ่งกีดขวางใน map (>=60) หรือใน dynamic/inflated mask
        static_ok = (v == -1) or (v < 60)
        dynamic_ok = True
        if self.inflated_mask is not None:
            dynamic_ok = not self.inflated_mask[y, x]
        return static_ok and dynamic_ok

    # -------------------- Heuristic --------------------
    def heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # -------------------- Dynamic mask builder --------------------
    def rebuild_dynamic_mask(self):
        """วาดสิ่งกีดขวางจาก LaserScan (ที่กรองระยะแล้ว) ลงบนกริด map และทำ inflation"""
        if not (self.map_ready and self.scan_last and self.update_pose()):
            return

        angle_min, angle_inc, ranges, rmin_eff, rmax_eff = self.scan_last
        H, W = self.height, self.width

        # เคลียร์มาสก์เดิม
        self.dynamic_mask[:] = False

        # downsample เพื่อประหยัดเวลา
        step = max(2, int(0.5 / max(angle_inc, 1e-4)))  # ~ทุก 0.5 rad
        # จำกัดมุมด้านหน้า
        front_half_span = math.radians(self.front_arc_deg) / 2.0

        # world pose
        xw, yw, yaw = self.pose
        R = rotate2d(yaw)

        for i in range(0, len(ranges), step):
            r = ranges[i]
            if r is None:
                continue  # ถูกกรองทิ้งเพราะอยู่นอกช่วงระยะแล้ว
            ang = angle_min + i * angle_inc
            if abs(ang) > front_half_span:
                continue

            # จุดใน local (base_link)
            lx = r * math.cos(ang)
            ly = r * math.sin(ang)
            # แปลงเป็น world
            wx, wy = (R @ np.array([lx, ly])) + np.array([xw, yw])
            gx, gy = self.world_to_grid((wx, wy))
            if 0 <= gx < W and 0 <= gy < H:
                self.dynamic_mask[gy, gx] = True

        # Inflation
        self.inflated_mask[:] = self.inflation(self.dynamic_mask, self.inflation_radius_m)

    def inflation(self, mask: np.ndarray, radius_m: float) -> np.ndarray:
        """ขยายรัศมีรอบ True cells โดยประมาณแบบวัดระยะ Chebyshev"""
        if mask is None or self.res is None:
            return mask
        rad_cells = max(1, int(radius_m / self.res))
        if rad_cells <= 0:
            return mask.copy()
        H, W = mask.shape
        out = mask.copy()
        rr = np.arange(-rad_cells, rad_cells + 1)
        XX, YY = np.meshgrid(rr, rr, indexing='xy')
        circle = (XX*XX + YY*YY) <= (rad_cells * rad_cells)
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            y0 = max(0, y - rad_cells); y1 = min(H, y + rad_cells + 1)
            x0 = max(0, x - rad_cells); x1 = min(W, x + rad_cells + 1)
            cy0 = rad_cells - (y - y0)
            cx0 = rad_cells - (x - x0)
            sub = circle[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)]
            out[y0:y1, x0:x1] |= sub
        return out

    # -------------------- A* --------------------
    def plan_path(self, custom_goal_world: np.ndarray = None):
        if not self.map_ready or not self.update_pose():
            return
        # อัปเดต dynamic mask จาก Lidar ก่อนวางแผน (ใช้ระยะที่กรองแล้ว)
        self.rebuild_dynamic_mask()

        self.is_replanning = True

        if custom_goal_world is None:
            forward = rotate2d(self.pose[2]).dot(np.array([self.goal_distance, 0.0]))
            goal_world = self.pose[:2] + forward
        else:
            goal_world = custom_goal_world

        start = self.world_to_grid(self.pose[:2])
        goal = self.world_to_grid(goal_world)

        # ถ้าเป้าหมายโดนบล็อก ลองขยับรอบๆ
        if not self.is_free(goal):
            self.get_logger().warn("⚠️ Goal cell blocked! Searching nearby...")
            found = False
            for r in range(1, 6):
                for dx, dy in [(r,0),(-r,0),(0,r),(0,-r),(r,r),(-r,-r),(r,-r),(-r,r)]:
                    g2 = (goal[0]+dx, goal[1]+dy)
                    if self.is_free(g2):
                        goal = g2
                        found = True
                        break
                if found:
                    break

        self.get_logger().info(f"🧭 Planning A* from {start} → {goal}")
        open_list = [(0, start)]
        came_from = {}
        g = {start: 0.0}
        f = {start: self.heuristic(start, goal)}
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        visited = set()

        while open_list:
            _, curr = heapq.heappop(open_list)
            if curr in visited:
                continue
            visited.add(curr)

            if curr == goal:
                path = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                self.path = [self.grid_to_world(p) for p in path]
                self.path_idx = 0
                self.path_ready = True
                self.is_replanning = False
                self.get_logger().info(f"✅ Path found: {len(self.path)} waypoints")
                return

            # Expand neighbors
            for dx, dy in dirs:
                nxt = (curr[0]+dx, curr[1]+dy)
                if not self.is_free(nxt):
                    continue
                ng = g[curr] + (1.4142 if dx != 0 and dy != 0 else 1.0)  # 8-connected
                if ng < g.get(nxt, float('inf')):
                    came_from[nxt] = curr
                    g[nxt] = ng
                    f[nxt] = ng + self.heuristic(nxt, goal)
                    heapq.heappush(open_list, (f[nxt], nxt))

        self.path_ready = False
        self.is_replanning = False
        self.get_logger().warn("❌ No path found!")

    # -------------------- Follow Path --------------------
    def follow_path(self):
        if not self.path_ready or self.path_idx >= len(self.path):
            return Twist(), True

        target = self.path[self.path_idx]
        dx, dy = target[0] - self.pose[0], target[1] - self.pose[1]
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        yaw_err = (ang - self.pose[2] + math.pi) % (2*math.pi) - math.pi

        # ปรับความเร็วเชิงเส้นตามระยะสิ่งกีดขวางด้านหน้า (soft slowdown)
        safe_lin = self.linear_speed
        if self.front_min_dist < 1.0:
            d = max(self.front_min_dist - self.emergency_stop_dist, 0.0)
            scale = min(max(d / (1.0 - self.emergency_stop_dist), 0.0), 1.0)
            safe_lin = max(0.05, self.linear_speed * scale)

        cmd = Twist()
        if abs(yaw_err) > self.angle_tol:
            cmd.angular.z = self.angular_speed * max(min(yaw_err, 1.0), -1.0)
        elif dist > self.dist_tol:
            cmd.linear.x = safe_lin
        else:
            self.path_idx += 1

        return cmd, self.path_idx >= len(self.path)

    # -------------------- Side-goal helper --------------------
    def compute_side_goals(self):
        """สร้างเป้าหมายเยื้องซ้าย/ขวา (เช่น 0.5 m) แล้วยิงไปข้างหน้า 3 m เพื่อหลบสิ่งกีดขวางใหญ่ด้านหน้า"""
        offs = [0.5, -0.5, 0.8, -0.8]
        goals = []
        R = rotate2d(self.pose[2])
        for lat in offs:
            local = np.array([self.goal_distance, lat])
            goals.append(self.pose[:2] + (R @ local))
        return goals

    # -------------------- Main Loop --------------------
    def loop(self):
        if not self.map_ready or not self.update_pose():
            return

        now = time.time()

        # 🚨 Emergency stop
        if self.front_min_dist < self.emergency_stop_dist:
            self.cmd_pub.publish(Twist())
            self.path_ready = False
            if (now - self.last_obstacle_time) > self.replan_cooldown:
                self.last_obstacle_time = now
                self.plan_path()  # replan กับ dynamic mask
            return

        # 🚧 Obstacle ahead → replan หลัง cooldown
        if self.front_min_dist < self.obstacle_threshold and not self.is_replanning:
            if now - self.last_obstacle_time > self.replan_cooldown:
                self.get_logger().warn(f"🛑 Obstacle {self.front_min_dist:.2f} m ahead → Replanning")
                self.path_ready = False
                self.last_obstacle_time = now
                self.plan_path()
                return

        # 🗺️ ถ้ายังไม่มี path → วางแผนใหม่
        if not self.path_ready and not self.is_replanning:
            self.plan_path()
            if not self.path_ready:
                for g in self.compute_side_goals():
                    self.get_logger().info(f"↪️ Try side goal at {g}")
                    self.plan_path(custom_goal_world=g)
                    if self.path_ready:
                        break
            return

        # 🚗 เคลื่อนตาม path
        cmd, done = self.follow_path()
        self.cmd_pub.publish(cmd)
        if done:
            self.cmd_pub.publish(Twist())
            self.get_logger().info("🏁 Goal reached!")
            self.path_ready = False

def main(args=None):
    rclpy.init(args=args)
    node = AStarTurtleBot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
