# Load required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pipeline import Pipeline
import os

# Pipeline
pipeline = Pipeline('sobel5', diag=True)

count = 0
# hist = False
# Get each file and process each
file = 'harder_challenge_video.mp4_snapshot_00.06_[2018.10.02_06.20.41].jpg'
file = 'harder_challenge_video.mp4_snapshot_00.34_[2018.10.02_11.42.14].jpg'
#file = 'straight_lines1.jpg'

# read in the image
image = cv2.imread('test_images/' + file)

pipeline.process_image(image)

#     if hist:
#         output, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = lane.process(image, 'poly_search', left_fit, right_fit)
#         plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
#         plt.show()

#         print('poly')
#         print('left_fit, right_fit', left_fit, right_fit)
#         print('left_curverad_real, right_curverad_real', left_curverad_real, right_curverad_real)
#         print('avg curve', np.mean([left_curverad_real, right_curverad_real]))

#         print('This image', file,' is:', type(image), 'with dimensions:', image.shape)

#         # write to image file
#         cv2.imwrite('output_images/poly_'+file,output)

output, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = pipeline.process(
    image)
# plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
# plt.show()
# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
print('hist')
print('left_fit, right_fit', left_fit, right_fit)
print('left_curverad_real, right_curverad_real', left_curverad_real, right_curverad_real)
print('avg curve', np.mean([left_curverad_real, right_curverad_real]))

# write to image file
cv2.imwrite('output_images/' + file, output)
