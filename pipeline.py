import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from sobelthreshold import SobelThreshold
from colorthreshold import ColorThreshold
from histogramsliding import HistogramSliding
from convolutionsliding import ConvolutionSliding
from profiles import profiles

# Pipeline
class Pipeline:
    def __init__(self, profile_name='default', diag = False):
        self.mtx, self.dist = self.load_distortion()
        self.profile = profiles[profile_name]

        self.lane_length_px = 700

        # Store all the procesed images
        self.undistorted_image = None
        self.sobel_filtered = None
        self.color_filtered = None
        self.filtered_image = None
        self.warped_image = None
        self.warped_filtered_image = None
        self.Minv = None

        self.diag = diag

        self.sliding_window_image = None

    def load_distortion(self):
        """
        Retrieve distortion coefficients from pickle file
        """
        dist_pickle = pickle.load(open("distortion_pickle.p", "rb"))
        mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']
        return mtx, dist

    def undistort_img(self, img, mtx, dist):
        """
        Undistort image
        """
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def warp(self, image):
        # Define the perspective transform source and destination points
        # src_top_left = (533, 490)
        # src_top_right = (752, 490)
        # src_bottom_right = (1053, 677)
        # src_bottom_left = (258, 677)

        src_top_left = (578, 460)
        src_top_right = (703, 460)
        src_bottom_right = (1123, 720)
        src_bottom_left = (193, 720)

        dest_top_left = (300, 0)
        dest_top_right = (980, 0)
        dest_bottom_right = (980, 720)
        dest_bottom_left = (300, 720)

        src = np.float32([src_top_left, src_top_right, src_bottom_right, src_bottom_left])
        dest = np.float32([dest_top_left, dest_top_right, dest_bottom_right, dest_bottom_left])

        # vertices = np.array([[src_bottom_left,src_top_left, src_top_right, src_bottom_right]], dtype=np.int32)
        # pts = vertices.reshape((-1,1,2))
        # cv2.polylines(image,[pts],True,(0,0,255), 1)

        # cv2.circle(image, src_top_left, 5, (0, 0, 255), -1)
        # cv2.circle(image, src_top_right, 5, (0, 0, 255), -1)
        # cv2.circle(image, src_bottom_right, 5, (0, 0, 255), -1)
        # cv2.circle(image, src_bottom_left, 5, (0, 0, 255), -1)


        img_size = (image.shape[1], image.shape[0])

        M = cv2.getPerspectiveTransform(src, dest)
        # Use cv2.warpPerspective() to warp image to a top-down view
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

        # calculate inverse perspective matrix
        Minv = cv2.getPerspectiveTransform(dest, src)

        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
        # f.tight_layout()
        # if len(image.shape) > 2:
        #     ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # else:
        #     ax1.imshow(image)
        # ax1.set_title('Before', fontsize=20)
        # if len(image.shape) > 2:
        #     ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        # else:
        #     ax2.imshow(warped, cmap='gray')
        #
        # ax2.set_title('Warped', fontsize=20)

        return warped, Minv

    def hist(self, img):
        '''
        Plot out histogram of bottom half of image
        '''
        bottom_half = img[(img.shape[0] // 2):, :]  # img [y:y, x:x]

        histogram = np.sum(bottom_half, axis=0)

        return histogram

    def measure_curvature_pixels(self, image, left_fit, right_fit):
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature) #####
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** (3 / 2)) / np.absolute(
            (2 * left_fit[0]))

        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** (3 / 2)) / np.absolute(
            (2 * right_fit[0]))

        return left_curverad, right_curverad

    def measure_curvature_real(self, image, left_fit, right_fit):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        y_eval = y_eval * ym_per_pix

        # Calculation of R_curve (radius of curvature) #####
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** (3 / 2)) / np.absolute(
            (2 * left_fit_cr[0]))

        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** (3 / 2)) / np.absolute(
            (2 * right_fit_cr[0]))

        return left_curverad, right_curverad

    def measure_offset(self, image, left_fit, right_fit):
        leftx = left_fit[0] * image.shape[0] ** 2 + left_fit[1] * image.shape[0] + left_fit[2]
        rightx = right_fit[0] * image.shape[0] ** 2 + right_fit[1] * image.shape[0] + right_fit[2]

        midpoint = (rightx + leftx) / 2
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # print('midpoint', midpoint)
        # print('image centre', image.shape[1]/2)

        offset = ((image.shape[1] / 2) - midpoint) * xm_per_pix

        return offset

    def fit_poly(self, x, y, ploty):
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

    def search_around_poly(self, binary_warped, left_fit, right_fit):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        margin = 50

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_fitx_low = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] - margin
        left_fitx_high = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] + margin
        right_fitx_low = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] - margin
        right_fitx_high = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] + margin

        left_lane_inds = ((nonzerox >= left_fitx_low) & (nonzerox < left_fitx_high)).nonzero()[0]
        right_lane_inds = ((nonzerox >= right_fitx_low) & (nonzerox < right_fitx_high)).nonzero()[0]

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # Fit new polynomials for left and right
        left_line_detected, new_left_fit, left_fitx = self.fit_poly(leftx, lefty, ploty)
        right_line_detected, new_right_fit, right_fitx = self.fit_poly(rightx, righty, ploty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        ## Visualization ##

        # Plots the left and right polynomials on the lane lines
        pts = np.array((left_fitx, ploty), dtype=np.int32).T
        pts.reshape((-1, 1, 2))
        cv2.polylines(result, [pts], False, (0, 255, 255), 5)

        pts = np.array((right_fitx, ploty), dtype=np.int32).T
        pts.reshape((-1, 1, 2))
        cv2.polylines(result, [pts], False, (0, 255, 255), 5)

        return result, new_left_fit, new_right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected

    def draw_lane(self, warped, undist, image, left_fitx, right_fitx, Minv):
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result

    def process_image(self, image):
        # Undistort image
        self.undistorted_image = self.undistort_img(image, self.mtx, self.dist)

        # Apply sobel threshold
        sobel_threshold = SobelThreshold(self.profile['sobel'])
        self.sobel_filtered = sobel_threshold.apply_sobel(self.undistorted_image)

        # Apply color threshold
        color_threshold = ColorThreshold()
        self.color_filtered = color_threshold.apply_threshold(self.undistorted_image)

        # Merge sobel and color
        self.filtered_image = np.zeros_like(self.color_filtered)
        self.filtered_image[(self.color_filtered == 1) & (self.sobel_filtered == 1)] = 255
        # self.filtered_image[(self.color_filtered == 1)] = 255

        # Apply perspective transform
        self.warped_image, self.Minv = self.warp(image)

        self.warped_filtered_image, filtered_Minv = self.warp(self.filtered_image)
    def find_parallel_poly(self, image, fit, lane_side):
        if lane_side == 'left':
            fit[2] = fit[2] - self.lane_length_px
        else:
            fit[2] = fit[2] + self.lane_length_px
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

        return fit, fitx, False
    def find_lanes(self, image):
        leftx = lefty = rightx = righty = left_curverad_real = right_curverad_real = offset = None
        histogram_sliding = HistogramSliding()
        convolution_sliding = ConvolutionSliding()

        # Do histogram sliding window search
        out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected = histogram_sliding.fit_polynomial(
            self.warped_filtered_image)
        self.sliding_window_image = out_image

        # Check outputs
        if left_fit is None and right_fit is None:
            # Do convolution search on both lines
            out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected = convolution_sliding.fit_polynomial(
                self.warped_filtered_image)
            self.sliding_window_image = out_image
        elif left_fit is None:
            # Do convolution search on left line
            left_out_image, left_fit, left_fitx, leftx, lefty, left_line_detected = convolution_sliding.fit_polynomial(
                self.warped_filtered_image, 'left')
        elif right_fit is None:
            # Do convolution search on right line
            right_out_image, right_fit, right_fitx, rightx, righty, right_line_detected = convolution_sliding.fit_polynomial(
                self.warped_filtered_image, 'right')

        if left_fit is not None and right_fit is not None:
            # Find offset and curvatures
            left_curverad_real, right_curverad_real = self.measure_curvature_real(image, left_fit, right_fit)

            # Calculate offset using histogram poly
            offset = self.measure_offset(image, left_fit, right_fit)

        return out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset
    def find_poly(self, image, left_search_fit, right_search_fit):
        leftx = lefty = rightx = righty = left_curverad_real = right_curverad_real = offset= None

        out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected = self.search_around_poly(
            self.warped_filtered_image, left_search_fit, right_search_fit)
        self.sliding_window_image = out_image

        if left_fit is not None and right_fit is not None:
            # Find offset and curvatures
            left_curverad_real, right_curverad_real = self.measure_curvature_real(image, left_fit, right_fit)

            # Calculate offset using histogram poly
            offset = self.measure_offset(image, left_fit, right_fit)

        return out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset
    def add_text(self, output, search_lane_type, left_curverad_real, right_curverad_real, offset):
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(output, search_lane_type, (0, 50), font, 2, (255, 0, 0), 2, cv2.LINE_AA)

        if left_curverad_real is not None and right_curverad_real is not None:
            cv2.putText(output, 'Radius of curvature: {}m'.format(round(np.mean([left_curverad_real, right_curverad_real]), 2)),
                        (0, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(output, 'Radius of curvature not found', (0, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if offset is None:
            offset_text = "Offset not found"
        elif offset == 0:
            offset_text = 'Vehicle is centred'
        elif offset < 0:
            offset_text = 'Vehicle is {}m left of center '.format(round(np.absolute(offset), 4))
        elif offset > 0:
            offset_text = 'Vehicle is {}m right of center '.format(round(np.absolute(offset), 4))

        cv2.putText(output, offset_text, (0, 150), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return output
    def process(self, image, search_type = 'sliding', left_search_fit = None, right_search_fit = None, left_best_fit = None, right_best_fit = None):
        leftx = lefty = rightx = righty = left_curverad_real = right_curverad_real = offset = None

        # Find lines, curvature and offset
        if search_type == 'sliding':
            out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = self.find_lanes(
                image)
        else:
            # Poly search
            out_image, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset = self.find_poly(image, left_search_fit, right_search_fit, left_best_fit, right_best_fit)

        # Draw lane
        output = self.draw_output(image, left_fitx, right_fitx)

        output = self.add_text(output, search_type, left_curverad_real, right_curverad_real, offset)

        return output, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, left_line_detected, right_line_detected, left_curverad_real, right_curverad_real, offset
    def draw_best_fit(self, image, left_fit, right_fit):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        result = self.draw_output(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), left_fitx, right_fitx)

        # Find offset and curvatures
        left_curverad_real, right_curverad_real = self.measure_curvature_real(image, left_fit, right_fit)

        # Calculate offset using histogram poly
        offset = self.measure_offset(image, left_fit, right_fit)

        result = self.add_text(result, 'Best fit', left_curverad_real, right_curverad_real, offset)
        return result

    def draw_output(self, image, left_fitx, right_fitx):
        # Draw lane
        result = self.draw_lane(self.warped_filtered_image, self.undistorted_image, image, left_fitx, right_fitx, self.Minv)

        if self.diag == True:
            # debug output
            height = int(image.shape[0] * 2)
            width = int(image.shape[1] * 3)
            output = np.zeros((height, width, 3), np.uint8)

            h, w, d = image.shape

            # output lane
            output[0:0 + h, 0:w] = result
            # cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # cv2.cvtColor(convo_result, cv2.COLOR_BGR2RGB)

            # output line tracking
            output[0:0 + h, w:w * 2] = self.sliding_window_image

            # output warped filtered image
            output[0:0 + h, w * 2:w * 3] = np.stack((self.warped_filtered_image,) * 3, -1)

            # output sobel filtered image
            sobel_filtered_image = np.zeros_like(self.color_filtered)
            sobel_filtered_image[(self.sobel_filtered == 1)] = 255
            output[h:h + h, 0:w] = cv2.bitwise_and(image, image,
                                                   mask=sobel_filtered_image)  # np.stack((sobel_filtered,)*3, -1)

            # output color filtered image
            output[h:h + h, w:w * 2] = cv2.bitwise_and(image, image, mask=self.color_filtered)

            # output combined filtered image
            output[h:h + h, w * 2:w * 3] = cv2.bitwise_and(image, image, mask=self.filtered_image)

            # output text
            font = cv2.FONT_HERSHEY_DUPLEX

            # add text
            # cv2.putText(output, 'Sliding window', (w, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(output, 'Warped filtered', (2 * w, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(output, 'Sobel filtered', (0, h + 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(output, 'Color filtered', (w, h + 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(output, 'Combined filtered', (2 * w, h + 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

            return output
        else:
            # Draw sliding window image
            print('draw')
            newImageHeight = int(self.sliding_window_image.shape[0]*0.3)
            newImageWidth = int(self.sliding_window_image.shape[1]*0.3)
            sliding_window_image = cv2.resize(self.sliding_window_image, (newImageWidth, newImageHeight))
            print('sliding_window_image.shape',sliding_window_image.shape)
            print('result.shape',result.shape)
            print('newImageHeight, newImageWidth',newImageHeight, newImageWidth)
            # result[:newImageHeight, :result.shape[1]-newImageWidth, :] = sliding_window_image
            result[20:20+newImageHeight, result.shape[1]-newImageWidth-20:result.shape[1]-20] = sliding_window_image
            return result
