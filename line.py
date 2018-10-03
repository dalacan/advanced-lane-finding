import numpy as np

# Lane line tracking class
class Line():
    def __init__(self):
        # Number of iterations to retain
        self.n_retain = 8

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def add(self, fit, fitx, x, y, line_detected, curverad_real, offset):
        print('fit', fit)
        print('curverad_real', curverad_real)
        print('offset', offset)
        print('offset', offset)
        print('line_detected', line_detected)
        if fit is not None and curverad_real is not None and offset is not None and line_detected:
            self.radius_of_curvature = curverad_real
            self.line_base_pos = offset

            # Do sanity checks
            if len(self.current_fit) > 0:
                self.diffs = self.current_fit[-1] - fit

            if len(self.current_fit) > 0 and (abs(self.diffs[0]) > 0.0002 or abs(self.diffs[1]) > 0.07 or abs(self.diffs[2])) > 30:
                print('coefficients not good')
                self.detected = False
                self.remove()
            # Check offset
            elif abs(self.line_base_pos) > 0.5:
                print('offset not good')
                self.detected = False
                self.remove()
            else:
                print("pass sanity")
                # Passed sanity check
                self.detected = True
                self.current_fit.append(fit)
                self.recent_xfitted.append(fitx)
                self.allx = x
                self.ally = y

                # Remove last fit if number of fits exceed retention number
                if len(self.current_fit) > self.n_retain:
                    print('remove retention')
                    self.remove()

                # Recalculate best fit
                if len(self.current_fit) > 0:
                    self.best_fit = np.mean(self.current_fit, axis=0)
                else:
                    self.best_fit = None
        else:
            self.detected = False
            self.remove()
    def remove(self):
        if len(self.current_fit) > 0:
            # Remove oldest fit
            self.current_fit = self.current_fit[1:]
            self.recent_xfitted = self.recent_xfitted[1:]

        # After removing, check current_fit array and calculate best fit
        if len(self.current_fit) > 0:
            # Recalculate best fit
            self.best_fit = np.mean(self.current_fit, axis=0)
        else:
            self.best_fit = None