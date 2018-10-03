import numpy as np
import cv2
import matplotlib.pyplot as plt

# Color threshold class
class ColorThreshold:
    def hls(self, image):
        """
        Get the hue, lightness and saturation values of the image
        """
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        return h_channel, l_channel, s_channel

    def hsv(self, image):
        """
        Get the hue, saturation and value (brightness) values of the image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        return h_channel, s_channel, v_channel

    def rgb(self, image):
        """
        Get the red, green and blue values of the image
        """
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]

        return R, G, B

    def apply_color_threshold(self, color_channel, threshold=(0, 255)):
        """
        Applies a color threshold filter on a specific color channel
        """
        binary = np.zeros_like(color_channel)
        binary[(color_channel > threshold[0]) & (color_channel <= threshold[1])] = 1

        return binary

    def apply_threshold_rgb(self, image):
        R, G, B = self.rgb(image)

        r_threshold = self.apply_color_threshold(R, (200, 255))
        g_threshold = self.apply_color_threshold(G, (200, 255))
        b_threshold = self.apply_color_threshold(B, (200, 255))

        binary = np.zeros_like(R)
        binary[(r_threshold == 1) & (g_threshold == 1) & (b_threshold == 1)] = 1

        return binary

    def apply_threshold_hsv(self, image):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        H, S, V = self.hsv(image)

        h_threshold = self.apply_color_threshold(H, (15, 30))
        s_threshold = self.apply_color_threshold(S, (90, 255))
        v_threshold = self.apply_color_threshold(V, (190, 255))

        binary = np.zeros_like(H)
        binary[(h_threshold == 1) & (v_threshold == 1) & (s_threshold == 1)] = 1

        return binary

    def apply_threshold_hls(self, image):
        H, L, S = self.hls(image)

        h_threshold = self.apply_color_threshold(H, (60, 100))
        l_threshold = self.apply_color_threshold(L, (190, 255))
        s_threshold = self.apply_color_threshold(S, (90, 255))

        binary = np.zeros_like(H)
        binary[(h_threshold == 1) & (l_threshold == 1) & (s_threshold == 1)] = 1

        return binary

    def apply_threshold(self, image):
        R, G, B = self.rgb(image)
        H, S, V = self.hsv(image)
        H, L, S = self.hls(image)

        # For white marking, use the green channel
        g_threshold = self.apply_color_threshold(G, (210, 255))
        r_threshold = self.apply_color_threshold(R, (210, 255))
        b_threshold = self.apply_color_threshold(B, (210, 255))

        # For yellow lines, use a combination of HS
        s_threshold = self.apply_color_threshold(S, (90, 255))
        h_threshold = self.apply_color_threshold(H, (10, 25))

        v_threshold = self.apply_color_threshold(V, (230, 255))
        l_threshold = self.apply_color_threshold(L, (230, 255))


        binary = np.zeros_like(G)
        binary[((g_threshold == 1) & (r_threshold == 1) & (b_threshold == 1)) | ((s_threshold == 1) & (h_threshold == 1))] = 1

        # Preview output
        # f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(48, 18))
        # f.tight_layout()
        # ax1.imshow(g_threshold, cmap='gray')
        # ax1.set_title('G thresh', fontsize=40)
        # ax2.imshow(s_threshold, cmap='gray')
        # ax2.set_title('S thresh', fontsize=40)
        # ax3.imshow(h_threshold, cmap='gray')
        # ax3.set_title('H thresh', fontsize=40)
        # ax4.imshow(binary, cmap='gray')
        # ax4.set_title('Binary GSH', fontsize=40)
        #
        # ax5.imshow(r_threshold, cmap='gray')
        # ax5.set_title('R thresh', fontsize=40)
        # ax6.imshow(b_threshold, cmap='gray')
        # ax6.set_title('B thresh', fontsize=40)
        # ax7.imshow(l_threshold, cmap='gray')
        # ax7.set_title('L thresh', fontsize=40)
        # ax8.imshow(v_threshold, cmap='gray')
        # ax8.set_title('V thresh', fontsize=40)
        #
        # ax9.imshow(L_channel, cmap='gray')
        # ax9.set_title('L channel', fontsize=40)
        # ax10.imshow(A_channel, cmap='gray')
        # ax10.set_title('A channel', fontsize=40)
        # ax11.imshow(B_channel, cmap='gray')
        # ax11.set_title('B channel', fontsize=40)
        # ax12.imshow(V, cmap='gray')
        # ax12.set_title('V channel', fontsize=40)
        #
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # plt.show()

        return binary


