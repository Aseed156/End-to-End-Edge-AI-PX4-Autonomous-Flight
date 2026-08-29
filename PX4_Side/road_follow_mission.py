#!/usr/bin/env python3
import math
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from uav_msgs.msg import RoadGeometry
from std_msgs.msg import Float32 as Float32Msg
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleStatus, VehicleAttitude
)


class State(Enum):
    INIT = auto()
    TAKEOFF = auto()
    FOLLOW = auto()
    LANDING = auto()
    DONE = auto()


class SubMode(Enum):
    VISION = auto()
    RECOVERY = auto()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_angle(a):
    """wrap to [-pi, pi]"""
    return math.atan2(math.sin(a), math.cos(a))


class PID:
    def __init__(self, kp, ki, kd, i_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False

    def reset(self):
        self.integral = 0.0
        self.has_prev = False

    def update(self, error, dt):
        self.integral = clamp(self.integral + error * dt, -self.i_limit, self.i_limit)
        derivative = (error - self.prev_error) / dt if (self.has_prev and dt > 0) else 0.0
        self.prev_error = error
        self.has_prev = True
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class RoadFollowMission(Node):
    def __init__(self):
        super().__init__('road_follow_mission')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- mission parameters ---
        self.takeoff_alt = 20.0        
        self.forward_speed = 3.0
        self.min_speed = 1.0
        self.conf_gate = 0.35           
        self.max_yaw_rate = 0.6
        self.max_lat_vel = 1.5
        self.ASSUMED_FOV_DEG = 84.0
        self.tile_ground_width_m = None
        self._fallback_ground_width_m = 2 * self.takeoff_alt * math.tan(math.radians(self.ASSUMED_FOV_DEG / 2))
        self._warned_using_fallback_footprint = False
        self.yaw_pid = PID(kp=1.0, ki=0.05, kd=0.15, i_limit=0.5)
        self.lat_pid = PID(kp=0.6, ki=0.02, kd=0.1, i_limit=1.0)


        self.waypoints = [
         (2.6, 45.5),   # bend_1  (lat=47.397994, lon=8.546769)
        (86.5, 83.5),   # bend_2  (lat=47.398749, lon=8.547274)
        (155.2, 147.8),   # bend_4  (lat=47.399367, lon=8.548127)
        (-8.3, 33.6),   # bend_4  (lat=47.397896, lon=8.546611)
        ]
        self.wp_idx = 0  # index of the waypoint we are currently heading to

        # corridor / recovery tuning
        self.corridor_limit_m = 12.0       # cross-track distance that triggers RECOVERY
        self.corridor_reentry_m = 6.0      # must be back inside this to be eligible for VISION again
        self.vision_reentry_streak_needed = 8   # consecutive good vision frames required to re-enter VISION
        self.waypoint_radius_m = 6.0       # "arrived" radius for intermediate waypoints
        self.landing_radius_m = 4.0        # "arrived" radius for the final waypoint
        self.recovery_speed = 2.5          # m/s straight-line speed while in RECOVERY

        self.submode = SubMode.VISION
        self._vision_good_streak = 0

        self.state = State.INIT
        self.home_x = self.home_y = None
        self.cur_x = self.cur_y = self.cur_z = 0.0
        self.yaw = 0.0
        self.lateral_offset_m = 0.0
        self.heading_error_rad = 0.0
        self.confidence = 0.0
        self.geom_valid = False
        self.last_geom_recv_time = None
        self.nav_state = None
        self.arm_state = None
        self.setpoint_counter = 0
        self.last_time = self.get_clock().now()

        self.create_subscription(RoadGeometry, '/jetson/road_geometry', self.road_cb, qos)
        self.create_subscription(Float32Msg, '/jetson/camera/footprint_m', self.footprint_cb, qos)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_pos_cb, qos)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_cb, qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos)

        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.timer = self.create_timer(0.05, self.loop)  # 20 Hz
        self.get_logger().info(
            f'road_follow_mission started, {len(self.waypoints)} waypoints loaded '
            f'(final = landing position)')

    # ---- callbacks ----
    def footprint_cb(self, msg):
        self.tile_ground_width_m = float(msg.data)

    def road_cb(self, msg):
        self.geom_valid = bool(msg.valid)
        self.confidence = float(msg.confidence)
        self.heading_error_rad = math.radians(float(msg.heading_error))

        if self.tile_ground_width_m is not None:
            ground_width_m = self.tile_ground_width_m
        else:
            ground_width_m = self._fallback_ground_width_m
            if not self._warned_using_fallback_footprint:
                self.get_logger().warn(
                    'no /jetson/camera/footprint_m received yet — using computed '
                    f'fallback footprint ({ground_width_m:.1f}m) for lateral_error conversion')
                self._warned_using_fallback_footprint = True

        self.lateral_offset_m = float(msg.lateral_error) * (ground_width_m / 2.0)
        self.last_geom_recv_time = self.get_clock().now()

    def local_pos_cb(self, msg):
        if self.home_x is None:
            self.home_x, self.home_y = msg.x, msg.y
            self.get_logger().info(f'home latched: x={self.home_x:.2f} y={self.home_y:.2f}')
        self.cur_x, self.cur_y, self.cur_z = msg.x, msg.y, msg.z

    def att_cb(self, msg):
        q = msg.q  # [w, x, y, z]
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state

    # ---- helpers ----
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)

    def send_offboard_heartbeat(self, velocity_mode):
        ocm = OffboardControlMode()
        ocm.position = not velocity_mode
        ocm.velocity = velocity_mode
        ocm.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(ocm)

    def geometry_is_fresh(self):
        if self.last_geom_recv_time is None:
            return False
        age = (self.get_clock().now() - self.last_geom_recv_time).nanoseconds / 1e9
        return age < 1.0

    def vision_is_good(self):
        return self.geometry_is_fresh() and self.geom_valid and self.confidence >= self.conf_gate

    def position_setpoint(self, x, y, z):
        sp = TrajectorySetpoint()
        sp.position = [x, y, z]
        sp.yaw = float('nan')
        sp.timestamp = self.get_clock().now().nanoseconds // 1000
        return sp

    def velocity_setpoint(self, vx, vy, vz, yaw_rate=0.0, yaw=None):
        sp = TrajectorySetpoint()
        sp.position = [float('nan'), float('nan'), float('nan')]
        sp.velocity = [vx, vy, vz]
        sp.acceleration = [float('nan'), float('nan'), float('nan')]
        if yaw is not None:
            sp.yaw = float(yaw)
            sp.yawspeed = float('nan')
        else:
            sp.yaw = float('nan')
            sp.yawspeed = yaw_rate
        sp.timestamp = self.get_clock().now().nanoseconds // 1000
        return sp

    # ---- waypoint / corridor geometry (local NED meters, absolute = home + offset) ----
    def wp_absolute(self, idx):
        ox, oy = self.waypoints[idx]
        return self.home_x + ox, self.home_y + oy

    def leg_endpoints(self):
        """Returns (start_xy, target_xy) for the current leg."""
        target = self.wp_absolute(self.wp_idx)
        if self.wp_idx == 0:
            start = (self.home_x, self.home_y)
        else:
            start = self.wp_absolute(self.wp_idx - 1)
        return start, target

    def cross_track_and_progress(self):
        (sx, sy), (tx, ty) = self.leg_endpoints()
        dx, dy = tx - sx, ty - sy
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return 0.0, 1.0
        t = ((self.cur_x - sx) * dx + (self.cur_y - sy) * dy) / seg_len_sq
        t_clamped = clamp(t, 0.0, 1.0)
        proj_x = sx + t_clamped * dx
        proj_y = sy + t_clamped * dy
        cross_track = math.hypot(self.cur_x - proj_x, self.cur_y - proj_y)
        return cross_track, t

    def distance_to_current_waypoint(self):
        tx, ty = self.wp_absolute(self.wp_idx)
        return math.hypot(self.cur_x - tx, self.cur_y - ty)

    def bearing_to_current_waypoint(self):
        tx, ty = self.wp_absolute(self.wp_idx)
        return math.atan2(ty - self.cur_y, tx - self.cur_x)

    def is_final_waypoint(self):
        return self.wp_idx >= len(self.waypoints) - 1

    # ---- main loop ----
    def loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.home_x is None:
            return  # wait for first local position fix

        if self.state == State.INIT:
            self.send_offboard_heartbeat(velocity_mode=False)
            self.setpoint_pub.publish(self.position_setpoint(self.home_x, self.home_y, -self.takeoff_alt))
            if self.setpoint_counter == 10:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)  # OFFBOARD
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)  # arm
                self.get_logger().info('sent arm + offboard, climbing to takeoff altitude')
                self.state = State.TAKEOFF
            self.setpoint_counter += 1
            return

        if self.state == State.TAKEOFF:
            self.send_offboard_heartbeat(velocity_mode=False)
            self.setpoint_pub.publish(self.position_setpoint(self.home_x, self.home_y, -self.takeoff_alt))
            if -self.cur_z > self.takeoff_alt - 1.0:
                self.get_logger().info(f'reached takeoff altitude ({-self.cur_z:.1f}m), starting road follow')
                self.yaw_pid.reset()
                self.lat_pid.reset()
                self.submode = SubMode.VISION
                self._vision_good_streak = 0
                self.state = State.FOLLOW
            return

        if self.state == State.FOLLOW:
            self.send_offboard_heartbeat(velocity_mode=True)

            # ---- advance waypoint index if arrived ----
            dist_to_wp = self.distance_to_current_waypoint()
            arrive_radius = self.landing_radius_m if self.is_final_waypoint() else self.waypoint_radius_m
            if dist_to_wp <= arrive_radius:
                if self.is_final_waypoint():
                    self.get_logger().info(
                        f'reached final waypoint ({dist_to_wp:.1f}m) - landing')
                    self.state = State.LANDING
                    return
                else:
                    self.wp_idx += 1
                    self.get_logger().info(
                        f'reached waypoint {self.wp_idx - 1}, advancing to waypoint {self.wp_idx}')
                    self.yaw_pid.reset()
                    self.lat_pid.reset()

            cross_track, progress = self.cross_track_and_progress()
            good_vision = self.vision_is_good()

            # ---- submode transition logic ----
            if self.submode == SubMode.VISION:
                if cross_track > self.corridor_limit_m or not good_vision:
                    self.submode = SubMode.RECOVERY
                    self._vision_good_streak = 0
                    self.get_logger().warn(
                        f'VISION -> RECOVERY (cross_track={cross_track:.1f}m, '
                        f'good_vision={good_vision}) heading to waypoint {self.wp_idx}')
            else:  # RECOVERY
                if good_vision:
                    self._vision_good_streak += 1
                else:
                    self._vision_good_streak = 0
                if cross_track < self.corridor_reentry_m and \
                        self._vision_good_streak >= self.vision_reentry_streak_needed:
                    self.submode = SubMode.VISION
                    self.yaw_pid.reset()
                    self.lat_pid.reset()
                    self.get_logger().info(
                        f'RECOVERY -> VISION (cross_track={cross_track:.1f}m, back on corridor)')

            # ---- control ----
            if self.submode == SubMode.VISION:
                if self.confidence < self.conf_gate:
                    fwd, yaw_rate, lat_vel = self.min_speed, 0.0, 0.0
                    self.yaw_pid.reset()
                    self.lat_pid.reset()
                else:
                    fwd = self.forward_speed
                    yaw_rate = clamp(self.yaw_pid.update(self.heading_error_rad, dt),
                                      -self.max_yaw_rate, self.max_yaw_rate)
                    lat_vel = clamp(-self.lat_pid.update(self.lateral_offset_m, dt),
                                     -self.max_lat_vel, self.max_lat_vel)

                vx_n = fwd * math.cos(self.yaw) - lat_vel * math.sin(self.yaw)
                vy_n = fwd * math.sin(self.yaw) + lat_vel * math.cos(self.yaw)
                self.setpoint_pub.publish(self.velocity_setpoint(vx_n, vy_n, 0.0, yaw_rate=yaw_rate))

                if self.setpoint_counter % 100 == 0:
                    self.get_logger().info(
                        f'[VISION] wp={self.wp_idx} dist_wp={dist_to_wp:.1f}m xtrack={cross_track:.1f}m '
                        f'lat_off={self.lateral_offset_m:.2f}m hdg_err={math.degrees(self.heading_error_rad):.1f}deg '
                        f'conf={self.confidence:.2f} nav_state={self.nav_state}')

            else:  # RECOVERY: fly straight at the waypoint, ignore vision
                bearing = self.bearing_to_current_waypoint()
                vx_n = self.recovery_speed * math.cos(bearing)
                vy_n = self.recovery_speed * math.sin(bearing)
                self.setpoint_pub.publish(self.velocity_setpoint(vx_n, vy_n, 0.0, yaw=bearing))

                if self.setpoint_counter % 100 == 0:
                    self.get_logger().info(
                        f'[RECOVERY] wp={self.wp_idx} dist_wp={dist_to_wp:.1f}m xtrack={cross_track:.1f}m '
                        f'good_streak={self._vision_good_streak} nav_state={self.nav_state}')

            self.setpoint_counter += 1
            return

        if self.state == State.LANDING:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info('landing commanded')
            self.state = State.DONE
            return

        if self.state == State.DONE:
            if self.setpoint_counter % 200 == 0:
                self.get_logger().info(f'mission complete (armed={self.arm_state}, nav_state={self.nav_state})')
            self.setpoint_counter += 1
            return


def main():
    rclpy.init()
    node = RoadFollowMission()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
