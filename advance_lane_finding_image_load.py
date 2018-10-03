# Load required libraries
import cv2
from pipeline import Pipeline
import os

# Pipeline
pipeline = Pipeline('sobel5')

# Get each file and process each
for file in os.listdir("test_images/"):
    # get file name and file extension
    file_name, file_ext = os.path.splitext(file)
    print("Processing image", file_name)
    # read in the image
    image = cv2.imread('test_images/' + file)

    # Do image conversion (undistort, warp and filter)
    pipeline.process_image(image)

    # Find lanes and plot lane
    output, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = pipeline.process(
        image)

    # write to image file
    cv2.imwrite('output_images/' + file, output)