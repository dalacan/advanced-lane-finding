import numpy as np
import cv2

# Sobel class
# Sobel class
class SobelThreshold:
    def __init__(self, sobel_profile):
        self.sobel_profile = sobel_profile
    def sobel_dir_thresh(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        """
        Applies sobel x and y, compute direction of gradient and apply threshold
        """
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        sobel_gradient = np.arctan2(abs_sobely, abs_sobelx)

        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(sobel_gradient)

        # 6) Return this mask as your binary_output image
        binary_output[((sobel_gradient >= thresh[0]) & (sobel_gradient <= thresh[1]))] = 1
        return binary_output

    def sobel_abs_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """
        Applies either sobel x or y and applies threshold on the absolute gradient
        """
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1 if orient == 'x' else 0, 1 if orient == 'y' else 0, ksize=sobel_kernel)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    def sobel_mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        """
        Compute magnitude of gradient from sobel x and y and apply threshold
        """
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Calculate the magnitude
        abs_sobelxy = np.sqrt((np.square(sobelx) + np.square(sobely)))

        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobelxy = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled_sobelxy)

        # 6) Return this mask as your binary_output image
        binary_output[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1

        return binary_output

    def apply_sobel(self, image):
        """
        Apply each of the sobel functions and combined them
        """
        # gradx = self.sobel_abs_thresh(image, orient='x', sobel_kernel=self.sobel_profile['sobel_abs_thresh_x']['sobel_kernel'], thresh=self.sobel_profile['sobel_abs_thresh_x']['threshold'])
        # grady = self.sobel_abs_thresh(image, orient='y', sobel_kernel=self.sobel_profile['sobel_abs_thresh_y']['sobel_kernel'], thresh=self.sobel_profile['sobel_abs_thresh_y']['threshold'])
        mag_binary = self.sobel_mag_thresh(image, sobel_kernel=self.sobel_profile['sobel_mag_thresh']['sobel_kernel'], mag_thresh=self.sobel_profile['sobel_mag_thresh']['threshold'])
        dir_binary = self.sobel_dir_thresh(image, sobel_kernel=self.sobel_profile['sobel_dir_thresh']['sobel_kernel'], thresh=self.sobel_profile['sobel_dir_thresh']['threshold'])

        combined = np.zeros_like(dir_binary)
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        combined[((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined
