import numpy as np
import cv2
import matplotlib.pyplot as plt

from line import Line
from pipeline import Pipeline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

pipeline = Pipeline('sobel5', diag=True)

left_line = Line()
right_line = Line()
lane_length_px = 700

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Load image
    pipeline.process_image(image)

    if (left_line.detected == False and right_line.detected == False) and (left_line.best_fit is None or right_line.best_fit is None):
        # Initial state
        print("Sliding #1")
        result, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = pipeline.process(image)

        left_line.add(left_fit, left_fitx, leftx, lefty, left_line_detected, left_curverad_real, offset)
        right_line.add(right_fit, right_fitx, rightx, righty, right_line_detected, right_curverad_real, offset)
    else:
        if (len(left_line.current_fit) > 0) and (len(right_line.current_fit) > 0):
             # Do poly search
            print("Poly search")
            out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = pipeline.find_poly(image, left_line.current_fit[-1], right_line.current_fit[-1])

            left_line.add(left_fit, left_fitx, leftx, lefty, left_line_detected, left_curverad_real, offset)
            right_line.add(right_fit, right_fitx, rightx, righty, right_line_detected, right_curverad_real, offset)

            left_best_fit = left_line.best_fit
            right_best_fit = right_line.best_fit

            # If best fit exists, draw best fit
            if left_best_fit is not None and right_best_fit is not None:
                # Draw best fit
                print("Best fit")
                result = pipeline.draw_best_fit(image, left_best_fit, right_best_fit)
            # Else use sliding window
            else:
                print("Sliding #2")
                result, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = pipeline.process(
                    image)

                left_line.add(left_fit, left_fitx, leftx, lefty, left_line_detected, left_curverad_real, offset)
                right_line.add(right_fit, right_fitx, rightx, righty, right_line_detected, right_curverad_real, offset)

        else:
            print("Sliding #3")
            result, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = pipeline.process(
                image)

            left_line.add(left_fit, left_fitx, leftx, lefty, left_line_detected, left_curverad_real, offset)
            right_line.add(right_fit, right_fitx, rightx, righty, right_line_detected, right_curverad_real, offset)


    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


### Process Video
# project_video_output = 'test_videos_output/project_video.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# project_video_clip = clip1.fl_image(process_image)
# project_video_clip.write_videofile(project_video_output, audio=False)

challenge_video_output = 'test_videos_output/challenge_video.mp4'
clip2 = VideoFileClip("challenge_video.mp4")
challenge_video_clip = clip2.fl_image(process_image)
challenge_video_clip.write_videofile(challenge_video_output, audio=False)
#
# harder_challenge_video_output = 'test_videos_output/harder_challenge_video.mp4'
# clip3 = VideoFileClip("harder_challenge_video.mp4")
# harder_challenge_video_clip = clip3.fl_image(process_image) #NOTE: this function expects color images!!
# harder_challenge_video_clip.write_videofile(harder_challenge_video_output, audio=False)