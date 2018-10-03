profiles = {
    'sobel5': {
        'sobel': {
            'sobel_dir_thresh': {
                'sobel_kernel': 15,
                'threshold': (0.7, 1.3)
            },
            'sobel_abs_thresh_x': {
                'sobel_kernel': 5,
                'threshold': (20, 100)
            },
            'sobel_abs_thresh_y': {
                'sobel_kernel': 5,
                'threshold': (20, 100)
            },
            'sobel_mag_thresh': {
                'sobel_kernel': 5,
                'threshold': (10, 50)
            }
        }
    },
    'default': {
        'sobel': {
            'sobel_dir_thresh': {
                'sobel_kernel': 15,
                'threshold': (0.7, 1.3)
            },
            'sobel_abs_thresh_x': {
                'sobel_kernel': 3,
                'threshold': (20, 100)
            },
            'sobel_abs_thresh_y': {
                'sobel_kernel': 3,
                'threshold': (20, 100)
            },
            'sobel_mag_thresh': {
                'sobel_kernel': 3,
                'threshold': (30, 100)
            }
        }
    }
}