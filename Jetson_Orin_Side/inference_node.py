#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

from jetson_vision.road_extractor import extract_road_information
from uav_msgs.msg import RoadGeometry


ENGINE_PATH = "/home/umer/Downloads/ddrnet23_slim.engine"
NUM_CLASSES = 2
INPUT_SIZE = 1024
ROAD_CLASS_ID = 1  # class index representing "road" in the softmax output

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine at {engine_path}")

        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
        output_shape = (1, NUM_CLASSES, INPUT_SIZE, INPUT_SIZE)

        self.h_input = cuda.pagelocked_empty(int(np.prod(input_shape)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(int(np.prod(output_shape)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.output_shape = output_shape

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        np.copyto(self.h_input, input_array.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.reshape(self.output_shape)


def preprocess(bgr_image: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return np.ascontiguousarray(img)


def postprocess_mask(raw_output: np.ndarray) -> np.ndarray:
    class_map = np.argmax(raw_output[0], axis=0)
    mask = np.where(class_map == ROAD_CLASS_ID, 255, 0).astype(np.uint8)
    return mask


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")

        self.get_logger().info("Loading TensorRT engine and allocating CUDA buffers...")
        t0 = time.time()
        self.trt_engine = TRTEngine(ENGINE_PATH)
        self.get_logger().info(f"Engine ready in {time.time() - t0:.2f}s")

        self.bridge = CvBridge()
        self._warmup()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.mask_pub = self.create_publisher(Image, "/jetson/road_mask", qos)
        self.geometry_pub = self.create_publisher(RoadGeometry, "/jetson/road_geometry", qos)
        self.image_sub = self.create_subscription(
            CompressedImage, "/laptop/sat_image", self.image_callback, qos
        )

        self.get_logger().info("inference_node ready, waiting for images.")

    def _warmup(self):
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for _ in range(2):
            self.trt_engine.infer(dummy)
        self.get_logger().info("Warmup complete.")

    def image_callback(self, msg: CompressedImage):
        t_start = time.time()

        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_image is None:
            self.get_logger().warn("Failed to decode incoming image, skipping frame.")
            return

        input_tensor = preprocess(bgr_image)
        raw_output = self.trt_engine.infer(input_tensor)
        mask = postprocess_mask(raw_output)

        out_header = Header()
        out_header.stamp = msg.header.stamp
        out_header.frame_id = "jetson_inference"

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = out_header
        self.mask_pub.publish(mask_msg)

        result = extract_road_information(mask)

        geo_msg = RoadGeometry()
        geo_msg.header = out_header
        geo_msg.valid = bool(result["valid"])
        geo_msg.lateral_error = float(result["lateral_error"])
        geo_msg.heading_error = float(result["heading_error"])
        geo_msg.confidence = float(result["confidence"])
        geo_msg.road_width = float(result["road_width"])
        geo_msg.road_center_x = float(result["road_center_x"])
        geo_msg.image_center_x = float(result["image_center_x"])
        geo_msg.coverage = float(result["coverage"])
        geo_msg.fit_quality = float(result["fit_quality"])
        geo_msg.solidity = float(result["solidity"])
        self.geometry_pub.publish(geo_msg)

        if not result["valid"]:
            self.get_logger().warn("No valid road geometry this frame — published valid=False.")

        latency_ms = (time.time() - t_start) * 1000
        self.get_logger().debug(f"Frame processed in {latency_ms:.1f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
