## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to identify the lane boundaries in a video. 

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

[//]: # (Image References)

[findchessboardoutput]: ./output_images/calibration8.jpg "Find chessboard output"
[undistorted]: ./output_images/undistorted.png "Undistored image"
[distortion_corrected_image]: ./output_images/distortion_corrected_image.png "Distortion corrected image"
[color_threshold]: ./output_images/color_threshold.png "Color threshold applied to image"
[sobel_threshold]: ./output_images/sobel_threshold.png "Gradient threshold applied to image"
[combined_filtered]: ./output_images/combined_filtered.png "Combined threshold filtering applied to image"
[warped_image]: ./output_images/warped_image.png "Warpped image"
[sliding_window_histogram]: ./output_images/sliding_window_histogram.png "Sliding windows histogram"
[sliding_window_convolution]: ./output_images/sliding_window_convolution.png "Sliding windows convolution"
[curvature_equation1]: ./output_images/curvature_equation1.PNG "Curvature equation"
[curvature_equation2]: ./output_images/curvature_equation2.PNG "Curvature equaation, 2nd order polynomial derived"
[straight_lines1_diagnostics]: ./output_images/straight_lines1_diagnostics.jpg "Output 1 with diagnostics"
[test4]: ./output_images/test4.jpg "Output results"
 
---

### Camera Calibration

#### Computation of the camera matrix and distortion coefficients

Prior to performing lane finding, the camera matrix and distortion coefficients will need to be calculated using chessboard images. The camera calibration code is contained within the `cameracalibration.py` file.

To begin, "object points", `objp` are initialized in 3D space with the x, y coordinates of corresponding to the chessboard image with z assumed to be 0. Next, I proceed to find the corners within the chessboard by 
1. Reading in a chessboard image, 
2. Converting the image into grayscale
3. Using the `cv2.findChessboardCorners()` function to find the image point corners within the image. *Note:* The chessboard size used for calibration is a 9 x 6.
![alt text][findchessboardoutput]
4. Check if corners were found. If they are found, I append it to my `imgpoints` list and append a corresponding `objpoints` from the `objp`.

The above steps are iterated through all the chessboard images and stored within the same `imgpoints` and `objpoints` list.

Next I use the `cv2.calibrateCamera` function to compute the camera matrix and distortion coefficient. Finally, the camera matrix and distortion is stored into a pickle file `distortion_pickle.p`. When needed, the camera matrix and distortion coefficient will be retrieved from the `distortion_pickle.p` file and used in-conjunction with an image and passed through the `cv2.undistort()` function. Below is a sample result:
![alt text][undistorted]


### Pipeline (single images)

#### Examples of a distortion-corrected image.

As part of the lane finding pipeline, the first step is to correct the distortion in the image. Below is an example of an image that had distortion correction applied:
![alt text][distortion_corrected_image]

#### Colour transformation and gradient threshold filtering

In my pipeline, I have tried and implemented both color and gradient threshold to isolate the white and yellow lane line markers with varying success. I have implemented the color transform within the `colorthreshold.py` file and the gradient masking within the `sobelthreshold.py` file.

Within the `colorthreshold.py`, I have implement several color transformation functions that can filter out colors for a specific channel and a given threshold range. The following color spaces are supported:
* RGB
* HSV
* HLS

Below is an example of color transformation applied each individual channel:
![alt text][color_threshold]

For my gradient transform, I've applied a combination of the magnitude and gradient direction masking. Below is an example of the gradient threshold being applied:

![alt text][sobel_threshold]

Once I've applied both transformation, I merged both color and gradient image into a single image whereby a perspective transform will be applied in the next step. Below is an example of the merged image.

![alt text][combined_filtered]

#### Perspective transformation

The next step in the lane finding process involves converting the image into a top down "bird's eye" view. To do so, I have created a function `pipeline.warp()` within the `pipeline.py` file. For my source points, I have manually choosen points from the `straight_lines1.jpg` image through visual inspection of the lane lines. Below is the before and after result after passing through the source and destination points into the `cv2.getPerspectiveTransform(src, dest)` function to first get the perspective transform matrix. This is followed up by warping the image through the `cv2.warpPerspective()` function. Additionally, ran the `cv2.getPerspectiveTransform(dest, src)` function to obtain the inverse perspective transform matrix which will be used later to project the lane highlighting back onto the road. Below is an example of a warped image with the red line highlighting the area of interest to transform.

![alt text][warped_image]


#### Lane line identification and polynomial fitting

To identify the lane pixel, I have implemented two types of sliding windows, with one using the histogram to find the x coordinates of the midpoint of the sliding window and the other using convolution whereby I sum up a window template with a vertical section of the image. Once I have identified the locations within the image whereby the lane is, I find all the x and y coordinates within each window that are lanes by cross referencing the windows against the image which are 1s. These sliding windows have been implemented in the following files `histogramsliding.py` and `convolutionsliding.py` respectively.

##### Sliding window - Histogram
![alt text][sliding_window_histogram]

##### Sliding window - Convolution
![alt text][sliding_window_convolution]

Next, using the found x and y coordinates, I attempt to find the 2nd order polynomial coefficients by passing the x and y coordinates through the `np.polyfit()` function.

Once I've found the polynomial coefficients, I get all the x and y coordinates by passing the y values,  y min (image top) to y max (image bottom), using the 2nd order polynomial formula f(y) = A y**2 + B y + C. These values will be used to draw the lane lines, find the radius of the curvature of the lane and position of the vehicle.

#### Calculating the radius of curvature of the lane and the position of the vehicle with respect to center

To calculate the radius of the curvature of the lane, I use the curvature formula:

![alt text][curvature_equation1]

As I am working with a 2nd order polynomial, I derive the curvature equation to:

![alt text][curvature_equation2]

By using the 2nd order coefficient values calculated previously, the radius of the curve can evaluated. Additionally, to find the real world radius, I use the following base on the assumption that a lane is 3.7 metres wide and 30 metres long in the image.

|  Dimension  |  metres per pixel  |
|-| -|
|  y  |  30/ 720  |
| x | 3.7 / 700 |

The calculation for the curvature can be found in the `pipeline.py` file under the `pipeline.measure_curvature_real()` function.

Next, to find the position of the vehicle I take the polynomial equation for both the left and right line and find the x coordinates at y max. Next I find the midpoint between the two x coordinates using the following formula:

> midpoint = (Xright + Xleft) / 2

Next, I evaluate the offset and the real world midpoint by subtracting the lane midpoint from the image midpoint (evaluated by image.shape[1] / 2) and multiplying the resultant by the pixel to real world scale for the x dimension. A positive offset value indicate that the car is right of the lane. Inversely, a negative offset indicates that the car is left of the lane.

> Offset = [(image.shape[1] / 2) - midpoint)] * xm_per_pix

The corresponding code can be found in th `pipeline.py` file under the `py.measure_offset()` function.

#### Results

To process images, I have implemented my pipeline through the `advance_lane_finding_image_load.py` file which will loop through all files in the test_image folder and process them. Below are a few results from the output of the `advance_lane_finding_image_load.py`.

##### Output with some diagnostics data
![alt text][straight_lines1_diagnostics]

##### Output
![alt text][test4]
---

### Pipeline (video)

#### Video results

My video pipeline is process through the `advance_lane_finding_video_load.py` file via the `process_image()` function and `line.py` file to track the left and right lines. In my video pipeline, I have implemented a basic filter which will use the previous frame polynomial coefficients to find the x position a the starting reference to find the lanes to find the new polynomial coefficient. From there I check the coefficient difference between the current and the previous frame and the lane offset to assess the degree of confidence in the polynomial coefficient. If the confidence is high (i.e. the coefficient variation is low and the lane offset value is < 0.5), I add the polynomial coefficient to a list of good polynomial coefficients for the past n frames (set to 8). To project a smooth line on the image, I use the good polynomial coefficients and then compute a mean coefficient which will be used to project a best fit line onto the image. If the confidence in the coefficient is low, I do not add it to the list. If the next n frame have low confidence, my function will revert back to using the sliding window calculation to find the lane pixels.

Here's a [link to my video result](./output_videos/project_video.mp4)

Here's a [link to my video result with a diagnostic breakdown of the lane finding](https://www.youtube.com/watch?v=gWUUoAdexzs "Advanced lane finding project video with diagnostics") [![link to my video result with a diagnostic breakdown](http://img.youtube.com/vi/gWUUoAdexzs/0.jpg)](https://www.youtube.com/watch?v=gWUUoAdexzs "Advanced lane finding project video with diagnostics")

---

### Reflections

Whilst trying to implement my pipeline, I encountered several hurdles in trying to identify lanes that are either poorly marked (i.e. a lot of black spots) or sharp corners. Ideally, I would have liked to implement a better curve lane finding pipeline which uses Hough line transformation and mean value theorem. To better identify poorly marked or hard to see lane marking, I would have liked to implement some form of shadow/illumination analysis.

