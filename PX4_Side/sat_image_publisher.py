#!/usr/bin/env python3
import math
import io
import threading
import cv2
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from px4_msgs.msg import VehicleGlobalPosition, VehicleAttitude, VehicleLocalPosition

TILE_SIZE = 256
ZOOM = 19  # ~0.3 m/px at equator
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
MAX_CACHED_TILES = 400
FETCH_WORKERS = 6


def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def meters_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


class TileCache:
    """Thread-safe LRU cache with background async fetch."""

    def __init__(self, max_tiles=MAX_CACHED_TILES, workers=FETCH_WORKERS):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._pending = set()
        self._max_tiles = max_tiles
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def get(self, x, y, z):
        key = (x, y, z)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if key in self._pending:
                return None
            self._pending.add(key)
        self._executor.submit(self._fetch, key)
        return None

    def _fetch(self, key):
        x, y, z = key
        try:
            url = TILE_URL.format(z=z, x=x, y=y)
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            img = PILImage.open(io.BytesIO(r.content)).convert('RGB')
            with self._lock:
                self._cache[key] = img
                self._cache.move_to_end(key)
                while len(self._cache) > self._max_tiles:
                    self._cache.popitem(last=False)
        except Exception:
            pass
        finally:
            with self._lock:
                self._pending.discard(key)


class SatImagePublisher(Node):
    def __init__(self):
        super().__init__('sat_image_publisher')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

        self.lat = self.lon = self.alt_rel = None
        self.yaw = 0.0
        self.assumed_fov_deg = 84.0
        self.tile_cache = TileCache()
        self.last_frame = None
        self.last_footprint_m = None
        self.frame_count = 0

        # Subscriptions
        self.create_subscription(VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.pos_cb, qos)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_pos_cb, qos)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_cb, qos)

        # Publishers aligned with system contract
        self.img_pub = self.create_publisher(CompressedImage, '/laptop/sat_image', qos)
        self.footprint_pub = self.create_publisher(Float32, '/jetson/camera/footprint_m', qos)

        self.timer = self.create_timer(0.05, self.publish_frame)  # 20 Hz matching requirements
        self.get_logger().info('sat_image_publisher started, waiting for position data...')

    def pos_cb(self, msg):
        if self.lat is None:
            self.get_logger().info(f'first global position received: lat={msg.lat:.6f} lon={msg.lon:.6f}')
        self.lat, self.lon = msg.lat, msg.lon

    def local_pos_cb(self, msg):
        if self.alt_rel is None:
            self.get_logger().info(f'first local position received: agl={-msg.z:.1f}m')
        self.alt_rel = -msg.z  # NED z is negative-up -> AGL height

    def att_cb(self, msg):
        q = msg.q  # [w, x, y, z]
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def stitch_around(self, lat, lon, zoom, grid=3):
        xt, yt = lonlat_to_tile(lon, lat, zoom)
        cx, cy = int(xt), int(yt)
        half = grid // 2
        canvas = PILImage.new('RGB', (TILE_SIZE * grid, TILE_SIZE * grid))
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                tile = self.tile_cache.get(cx + i, cy + j, zoom)
                if tile is None:
                    return None, None, None
                canvas.paste(tile, ((i + half) * TILE_SIZE, (j + half) * TILE_SIZE))
        px = (xt - (cx - half)) * TILE_SIZE
        py = (yt - (cy - half)) * TILE_SIZE
        return canvas, px, py

    def publish_frame(self):
        if self.lat is None or self.alt_rel is None:
            return

        canvas, px, py = self.stitch_around(self.lat, self.lon, ZOOM)

        if canvas is None:
            if self.last_frame is not None:
                self.img_pub.publish(self.last_frame)
            if self.last_footprint_m is not None:
                self.footprint_pub.publish(Float32(data=self.last_footprint_m))
            return

        mpp = meters_per_pixel(self.lat, ZOOM)
        ground_span_m = 2 * self.alt_rel * math.tan(math.radians(self.assumed_fov_deg) / 2)
        crop_px = max(32, int(ground_span_m / mpp))

        canvas_limit = int(TILE_SIZE * 3 / math.sqrt(2)) - 20
        if crop_px > canvas_limit:
            self.get_logger().warn(
                f'crop_px={crop_px} exceeds safe canvas limit={canvas_limit} '
                f'(alt_agl={self.alt_rel:.1f}m) - increase stitch grid or fly lower',
                throttle_duration_sec=5.0)
            crop_px = canvas_limit
        actual_footprint_m = crop_px * mpp

        left, upper = px - crop_px, py - crop_px
        right, lower = px + crop_px, py + crop_px
        cropped = canvas.crop((left, upper, right, lower))
        rotated = cropped.rotate(-math.degrees(self.yaw), resample=PILImage.BICUBIC)

        w, h = rotated.size
        final = rotated.crop((w // 2 - crop_px // 2, h // 2 - crop_px // 2,
                              w // 2 + crop_px // 2, h // 2 + crop_px // 2))
        final = final.resize((1024, 1024))

        # Convert to BGR array for JPEG encoding
        arr_bgr = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
        ret, jpeg_buf = cv2.imencode('.jpg', arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if not ret:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.format = 'jpeg'
        msg.data = jpeg_buf.tobytes()

        self.last_frame = msg
        self.last_footprint_m = actual_footprint_m
        self.img_pub.publish(msg)
        self.footprint_pub.publish(Float32(data=actual_footprint_m))

        self.frame_count += 1
        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f'published {self.frame_count} frames (crop_px={crop_px}, '
                f'footprint={actual_footprint_m:.1f}m, agl={self.alt_rel:.1f}m)')


def main():
    rclpy.init()
    node = SatImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
