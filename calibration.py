import pyzed.sl as sl
import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting calibration process...")

checkerboard_size = (7, 5)  # Number of inner corners per row and column
square_size = 0.029  # Size of a square in your defined unit (e.g., meters)

objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points from the camera

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

save_dir = "C:/project_data11/calibration_images1_zed"
os.makedirs(save_dir, exist_ok=True)

zed = sl.Camera()
init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080, camera_fps=30)

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    logging.error(f"Error: Unable to open ZED camera. Status: {status}")
    exit(1)

cv2.namedWindow('ZED Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ZED Calibration', 1920, 1080)  # Set this to your resolution

capture_count = 0
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()

        zed.retrieve_image(image, sl.VIEW.LEFT)

        frame = np.array(image.get_data(), dtype=np.uint8, copy=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display corners
            cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)
            cv2.imshow("Detected Corners", frame)
            cv2.waitKey(1)  # Display the frame for a short duration

            frame_path = os.path.join(save_dir, f"calib_{capture_count}.png")
            cv2.imwrite(frame_path, frame)
            logging.info(f"Captured calibration image {capture_count}: {frame_path}")
            capture_count += 1

        else:
            logging.warning("Chessboard not detected in the current frame.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting calibration loop...")
            break

cv2.destroyAllWindows()
zed.close()

logging.info("Performing intrinsic calibration...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

np.save(os.path.join(save_dir, "camera_matrix.npy"), camera_matrix)
np.save(os.path.join(save_dir, "dist_coeffs.npy"), dist_coeffs)

logging.info(f"Reprojection Error: {ret}")
logging.info("Intrinsic Calibration Completed Successfully.")

camera_matrix_loaded = np.load(os.path.join(save_dir, "camera_matrix.npy"))
dist_coeffs_loaded = np.load(os.path.join(save_dir, "dist_coeffs.npy"))
logging.info(f"Loaded Camera Matrix: {camera_matrix_loaded}")
logging.info(f"Loaded Distortion Coefficients: {dist_coeffs_loaded}")

