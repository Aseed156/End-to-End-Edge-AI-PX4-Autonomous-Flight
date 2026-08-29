# Vision-Based Autonomous UAV Road-Following System

An end-to-end pipeline for vision-guided autonomous drone navigation along road networks, built during an internship at **RESOLVE Research Center** (SUPARCO-affiliated).

The system combines real-time semantic segmentation with a simulated flight stack to let a drone detect a road beneath it and autonomously follow its centerline.

---

## Overview

- **Perception:** DDRNet-23-slim segmentation model, trained via sequential transfer learning and deployed as a TensorRT FP16 engine on a Jetson Orin Nano — **78% mIoU** on aerial imagery.
- **Flight control:** PX4 SITL + Gazebo Harmonic simulation, driven by a hybrid **VISION / RECOVERY** state machine mission controller.
- **Communication:** ROS2 (Humble) with CycloneDDS for cross-machine messaging between the flight-sim laptop and the Jetson.

## System Architecture

The pipeline spans two machines linked over USB-to-Ethernet:

| Machine | Role | Stack |
|---|---|---|
| Laptop (WSL2) | Flight simulation & mission control | PX4 SITL, Gazebo Harmonic, ROS2 Humble, QGroundControl |
| Jetson Orin Nano | Onboard AI inference | TensorRT FP16 engine, JetPack 6.2, CUDA 12.6 |

### Core Components

- `sat_image_publisher.py` — Synthesizes live satellite-tile imagery from real-time GPS/attitude data
- `inference_node.py` — DDRNet TensorRT inference node (FP16, pagelocked buffers, `execute_async_v3`, pycuda)
- `road_extractor.py` — Polynomial centerline fitting with a three-component confidence score (coverage × fit quality × solidity)
- `road_follow_mission.py` — Waypoint-corridor mission controller with a VISION/RECOVERY state machine & PID Controllers.
- `latlon_to_ned.py` — Coordinate converter for the PX4 SITL home fix

## Model Training

DDRNet-23-slim was fine-tuned through a three-stage transfer learning pipeline:

1. **Massachusetts Roads** dataset (~0.489 IoU)
2. **DeepGlobe** road extraction dataset (~0.613 IoU)
3. **Semantic Drone Dataset + Aeroscapes** fine-tuning :  **78% mIoU**

The trained model was exported to ONNX (opset 18) and compiled into a TensorRT FP16 engine directly on the Jetson, with no measurable accuracy drop from the PyTorch baseline.

## Simulation Stack

- PX4-Autopilot v1.15.4
- Gazebo Harmonic (headless, GPU-free)
- ROS2 Humble
- Micro-XRCE-DDS-Agent v2.4.2
- px4_msgs / px4_ros_com (release/1.15)

## Results

- **78% mIoU** semantic segmentation on aerial road imagery
- ~73.5% valid road detections in batch evaluation on Aeroscapes
- Fully integrated, closed-loop demo: satellite image synthesis → segmentation → centerline extraction → PX4 offboard control

## Acknowledgements

Developed during an internship at **RESOLVE Research Center**, under the supervision of **Dr. Shakeel Ur Rehman**.

---

*This repository documents a research prototype built for demonstration purposes; it is not flight-certified for real-world UAV operation.*
