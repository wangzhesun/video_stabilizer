import numpy as np
import cv2 as cv
from ssc import ssc


SMOOTHING_RADIUS = 30   # the greater, the smoother, the more likely of black border


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size  # Define the filter

    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')  # Add padding to the boundaries
    curve_smoothed = np.convolve(curve_pad, f, mode='same')  # Apply convolution
    curve_smoothed = curve_smoothed[radius:-radius]  # Remove padding
    return curve_smoothed  # return smoothed curve


def smooth(trajectory):
    new_trajectory = np.copy(trajectory)

    for i in range(3):  # Filter the x, y and angle curves
        new_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return new_trajectory


def fix_border(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    # T = cv.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    T = cv.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.2)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))
    return frame


def detect_feature(frame, feature_method):
    s = frame.shape
    if feature_method == 'o':
        orb_obj = cv.ORB_create()
        prev_pts = orb_obj.detect(frame, None)
        prev_pts = sorted(prev_pts, key=lambda x: x.response, reverse=True)
        prev_pts = ssc(prev_pts, 1000, 50, s[1], s[0])
        prev_pts = cv.KeyPoint_convert(prev_pts)
    elif feature_method == 's':
        sift_obj = cv.SIFT_create()
        prev_pts = sift_obj.detect(frame, None)
        prev_pts = ssc(prev_pts, 1000, 50, s[1], s[0])
        prev_pts = cv.KeyPoint_convert(prev_pts)
    elif feature_method == 'f':
        fast = cv.FastFeatureDetector_create()
        prev_pts = fast.detect(frame, None)
        prev_pts = cv.KeyPoint_convert(prev_pts)
    else:
        prev_pts = cv.goodFeaturesToTrack(frame, maxCorners=200, qualityLevel=0.01,
                                          minDistance=30, blockSize=3)

    return prev_pts


def transform(transforms_smooth, index, frame, w, h):
    # Extract transformations from the new transformation array
    dx = transforms_smooth[index, 0]
    dy = transforms_smooth[index, 1]
    da = transforms_smooth[index, 2]

    # Reconstruct transformation matrix accordingly to new values
    H = np.zeros((2, 3), np.float32)
    H[0][0] = np.cos(da)
    H[0][1] = -np.sin(da)
    H[1][0] = np.sin(da)
    H[1][1] = np.cos(da)
    H[0][2] = dx
    H[1][2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv.warpAffine(frame, H, (w, h))
    return fix_border(frame_stabilized)  # Fix border artifacts
