# Load required libraries
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import os

# prepare object points
nx = 9
ny = 6

# initialize object points
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

for image in images:
    # Read image
    img = cv2.imread(image)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        file = os.path.basename(image)
        cv2.imwrite('output_images/' + file, img)

# Calculate distortion coefficients and camera matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the distortion coefficients and camera matrix
dist_pickle = {
    "mtx": mtx,
    "dist": dist
}
pickle.dump(dist_pickle, open("distortion_pickle.p", "wb"))