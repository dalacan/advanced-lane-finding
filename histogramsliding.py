import numpy as np
import cv2
import matplotlib.pyplot as plt

# Finding lines using histogram sliding window
class HistogramSliding:
    def __init__(self):
        # HYPERPARAMETERS
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Choose the number of sliding windows
        self.nwindows = 9

    def find_hist_peak_x(self, binary_warped, y_min, y_max, lane_side):
        '''
        Find peak x from history for a specific y section
        '''
        # Take a histogram of the bottom half of the image (ymin:ymax, x:x)
        histogram = np.sum(binary_warped[y_min:y_max, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)

        if lane_side == 'left':
            x_base = np.argmax(histogram[:midpoint])
        else:
            x_base = np.argmax(histogram[midpoint:]) + midpoint

        return x_base

    def find_lane_pixels_window(self, binary_warped, nonzeroy, nonzerox, win_y_low, win_y_high, window, window_height,
                                x_current, lane_inds, lane_side):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # Find the four below boundaries of the window
        win_x_low = x_current - self.margin
        win_x_high = x_current + self.margin

        # Identify the nonzero pixels in x and y within the window ###
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                    nonzerox < win_x_high)).nonzero()[0]

        # Append these indices to the lists
        lane_inds.append(good_inds)

        # If found indice count > minpix pixels, recenter next window ###
        # (`right` or `leftx_current`) on their mean position ###
        if len(good_inds) > self.minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
        # elif lane_side == 'left':
        #     #             print('finding new x - left. Good left count: {} Window: {}'.format(len(good_inds), window))
        #     x_current = self.find_hist_peak_x(binary_warped, (win_y_low - window_height), (win_y_high - window_height),
        #                                       'left')
        # else:
        #     #             print('finding new x - right. Good right count: {} Window: {}'.format(len(good_inds), window))
        #     x_current = self.find_hist_peak_x(binary_warped, (win_y_low - window_height), (win_y_high - window_height),
        #                                       'right')

        return win_x_low, win_x_high, x_current, lane_inds

    def find_lane_pixels(self, binary_warped):
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base = self.find_hist_peak_x(binary_warped, (binary_warped.shape[0] // 2), binary_warped.shape[0], 'left')
        rightx_base = self.find_hist_peak_x(binary_warped, (binary_warped.shape[0] // 2), binary_warped.shape[0],
                                            'right')

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low, win_xleft_high, leftx_current, left_lane_inds = self.find_lane_pixels_window(binary_warped,
                                                                                                        nonzeroy,
                                                                                                        nonzerox,
                                                                                                        win_y_low,
                                                                                                        win_y_high,
                                                                                                        window,
                                                                                                        window_height,
                                                                                                        leftx_current,
                                                                                                        left_lane_inds,
                                                                                                        'left')

            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)

            win_xright_low, win_xright_high, rightx_current, right_lane_inds = self.find_lane_pixels_window(
                binary_warped, nonzeroy, nonzerox, win_y_low, win_y_high, window, window_height, rightx_current,
                right_lane_inds, 'right')
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def find_lane_pixels_lane(self, binary_warped, lane_side):
        """
        Find lane pixels for a specific lane
        """
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left or right halves of the histogram
        # These will be the starting point for the left or right lines
        x_base = self.find_hist_peak_x(binary_warped, (binary_warped.shape[0] // 2), binary_warped.shape[0], lane_side)

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current = x_base

        # Create empty lists to receivelane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_x_low, win_x_high, x_current, lane_inds = self.find_lane_pixels_window(binary_warped, nonzeroy,
                                                                                       nonzerox, win_y_low, win_y_high,
                                                                                       window, window_height, x_current,
                                                                                       lane_inds, lane_side)
            cv2.rectangle(out_img, (win_x_low, win_y_low),
                          (win_x_high, win_y_high), (0, 255, 0), 2)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        return x, y, out_img

    def fit_line(self, x, y, ploty):
        fit = None
        try:
            fit = np.polyfit(y, x, 2)
            fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
            line_detected = True
        except TypeError:
            # Catch error if fit are none or in correct
            fitx = 1 * ploty ** 2 + 1 * ploty
            line_detected = False

        return line_detected, fit, fitx

    def visualize(self, out_img, x, y, fitx, ploty, lane_side):
        """
        Add visualization of detected line to image
        """
        # Colors in the left and right lane regions
        if lane_side == 'left':
            out_img[y, x] = [255, 0, 0]
        else:
            out_img[y, x] = [0, 0, 255]

        pts = np.array((fitx, ploty),  dtype=np.int32).T
        pts.reshape((-1, 1, 2))
        cv2.polylines(out_img, [pts], False, (0, 255, 255), 5)

        # plt.imshow(out_img)
        # Plots the polynomials on the lane lines
        # plt.plot(fitx, ploty, color='yellow')
    def fit_polynomial(self, binary_warped, lane_side='both'):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # Find our lane pixels first
        if lane_side == 'both':
            # Find lane pixels
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)
            # Line polynomial coefficients and get fit
            left_line_detected, left_fit, left_fitx = self.fit_line(leftx, lefty, ploty)
            right_line_detected, right_fit, right_fitx = self.fit_line(rightx, righty, ploty)

            # visualization
            self.visualize(out_img, leftx, lefty, left_fitx, ploty, 'left')
            self.visualize(out_img, rightx, righty, right_fitx, ploty, 'right')
            # plt.show()

            return out_img, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected
        else:  # lane_side = right/right
            # Find lane pixels
            x, y, out_img = self.find_lane_pixels_lane(binary_warped, lane_side)
            # Line polynomial coefficients and get fit
            line_detected, fit, fitx = self.fit_line(x, y, ploty)

            # visualization
            # self.visualize(out_img, x, y, fitx, ploty, lane_side)
            # plt.show()

            return out_img, fit, fitx, x, y, line_detected