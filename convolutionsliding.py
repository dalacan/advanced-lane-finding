import numpy as np
import cv2
import matplotlib.pyplot as plt

# Finding lines using convolution sliding window
class ConvolutionSliding:
    def __init__(self):
        # window settings
        self.window_width = 50
        self.window_height = 80  # Break image into 9 vertical layers since image height is 720
        self.margin = 100  # How much to slide left and right for searching

    def window_mask(self, width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
        max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return output

    def window_indices(self, width, height, img_ref, center, level, nonzeroy, nonzerox):
        # y
        win_y_low = int(img_ref.shape[0] - (level + 1) * height)
        win_y_high = int(img_ref.shape[0] - level * height)

        # x
        win_x_low = max(0, int(center - width / 2))
        win_x_high = min(int(center + width / 2), img_ref.shape[1])

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                nonzerox < win_x_high)).nonzero()[0]

        return good_inds

    def find_window_centroids(self, image):
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - self.window_width / 2
        r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - self.window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(image.shape[0] / self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                image[
                int(image.shape[0] - (level + 1) * self.window_height):int(image.shape[0] - level * self.window_height),
                :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width / 2
            l_min_index = int(max(l_center + offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids

    def find_window_centroids_lane(self, image, lane_side):
        """
        Find centroids for a specific lane
        """
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        if lane_side == 'left':
            img_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
            center = np.argmax(np.convolve(window, img_sum)) - self.window_width / 2
        else:  # right
            img_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
            center = np.argmax(np.convolve(window, img_sum)) - self.window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(image.shape[0] / self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                image[
                int(image.shape[0] - (level + 1) * self.window_height):int(image.shape[0] - level * self.window_height),
                :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width / 2
            min_index = int(max(center + offset - self.margin, 0))
            max_index = int(min(center + offset + self.margin, image.shape[1]))
            center = np.argmax(conv_signal[min_index:max_index]) + min_index - offset

            # Add what we found for that layer
            window_centroids.append((center))

        return window_centroids

    def find_lane_pixels(self, warped, window_centroids):
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = self.window_mask(self.window_width, self.window_height, warped, window_centroids[level][0], level)
            r_mask = self.window_mask(self.window_width, self.window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

            good_left_inds = self.window_indices(self.window_width, self.window_height, warped,
                                                 window_centroids[level][0], level, nonzeroy, nonzerox)
            good_right_inds = self.window_indices(self.window_width, self.window_height, warped,
                                                  window_centroids[level][1], level, nonzeroy, nonzerox)

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

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

        return leftx, lefty, rightx, righty, l_points, r_points

    def find_lane_pixels_lane(self, warped, window_centroids):
        """
        Find lane pixels for a specific lane
        """
        # Create empty lists to receive left or right lane pixel indices
        lane_inds = []

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Points used to draw all the left or right windows
        points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            mask = self.window_mask(self.window_width, self.window_height, warped, window_centroids[level], level)
            # Add graphic points from window mask here to total pixels found
            points[(points == 255) | ((mask == 1))] = 255

            good_inds = self.window_indices(self.window_width, self.window_height, warped, window_centroids[level],
                                            level, nonzeroy, nonzerox)

            lane_inds.append(good_inds)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        return x, y, points

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

    def fit_polynomial(self, warped, lane_side='both'):
        if lane_side == 'both':
            window_centroids = self.find_window_centroids(warped)

            # If we found any window centers
            if len(window_centroids) > 0:
                # Generate x and y values for plotting
                ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

                leftx, lefty, rightx, righty, l_points, r_points = self.find_lane_pixels(warped, window_centroids)

                left_line_detected, left_fit, left_fitx = self.fit_line(leftx, lefty, ploty)
                right_line_detected, right_fit, right_fitx = self.fit_line(rightx, righty, ploty)




                # Draw the results
                template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
                zero_channel = np.zeros_like(template)  # create a zero color channel
                template = np.array(cv2.merge((zero_channel, template, zero_channel)),
                                    np.uint8)  # make window pixels green
                warpage = np.dstack((warped, warped, warped)) * 255  # making the original road pixels 3 color channels
                output = cv2.addWeighted(warpage, 1, template, 0.5,
                                         0.0)  # overlay the orignal road image with window results

                ## Visualization ##

                # Plots the left and right polynomials on the lane lines
                pts = np.array((left_fitx, ploty), dtype=np.int32).T
                pts.reshape((-1, 1, 2))
                cv2.polylines(output, [pts], False, (0, 255, 255), 5)

                pts = np.array((right_fitx, ploty), dtype=np.int32).T
                pts.reshape((-1, 1, 2))
                cv2.polylines(output, [pts], False, (0, 255, 255), 5)

            # If no window centers found, just display original road image
            else:
                output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

            # Display the final results
            # plt.imshow(output)
            # plt.title('window fitting results')
            # plt.show()
            return output, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected
        else:  # lane side is left or right
            window_centroids = self.find_window_centroids_lane(warped, lane_side)

            # If we found any window centers
            if len(window_centroids) > 0:
                # Generate x and y values for plotting
                ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

                x, y, points = self.find_lane_pixels_lane(warped, window_centroids)

                line_detected, fit, fitx = self.fit_line(x, y, ploty)

                ## Visualization ##

                # Plots the left and right polynomials on the lane lines
                # plt.plot(fitx, ploty, color='yellow')

                # Draw the results
                template = np.array(points, np.uint8)  # add both left and right window pixels together
                zero_channel = np.zeros_like(template)  # create a zero color channel
                template = np.array(cv2.merge((zero_channel, template, zero_channel)),
                                    np.uint8)  # make window pixels green
                warpage = np.dstack((warped, warped, warped)) * 255  # making the original road pixels 3 color channels
                output = cv2.addWeighted(warpage, 1, template, 0.5,
                                         0.0)  # overlay the orignal road image with window results

            # If no window centers found, just display original road image
            else:
                output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

            # Display the final results
            # plt.imshow(output)
            # plt.title('window fitting results')
            # plt.show()
            return output, fit, fitx, x, y, line_detected



