# ZED2-Camera-Calibration
Calibration script for stereo ZED 2 camera setup using OpenCV and Stereolabs SDK for accurate depth and 3D reconstruction.
# ZED 2 Camera Calibration using OpenCV and Stereolabs SDK

This project provides a Python script for **stereo camera calibration** of the ZED 2 camera using a checkerboard pattern. The goal is to compute intrinsic parameters (camera matrix and distortion coefficients) which are essential for 3D vision applications like depth estimation and stereo reconstruction.

---

## 📷 Setup

- **Camera**: ZED 2
- **Checkerboard Size**: 7 x 5 inner corners
- **Square Size**: 0.029 meters
- **Resolution**: HD1080 @ 30 FPS
- **Software**: 
  - Python
  - OpenCV
  - Stereolabs Python API (`pyzed`)

---

## 📁 Directory Structure

