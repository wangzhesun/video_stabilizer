import numpy as np
import cv2 as cv
from utils.ssc import ssc
from matplotlib import pyplot as plt

SMOOTHING_RADIUS = 30  # the greater, the smoother, the more likely of black border


def moving_average(curve, radius):
    """
    helper function for the smooth function

    :param curve: function curve to be smoothed
    :param radius: number of frames for the smoothing process
    :return: smoothed function curve
    """
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size  # Define the filter

    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')  # Add padding to the boundaries
    curve_smoothed = np.convolve(curve_pad, f, mode='same')  # Apply convolution
    curve_smoothed = curve_smoothed[radius:-radius]  # Remove padding
    return curve_smoothed  # return smoothed curve


def smooth(trajectory, live=0):
    """
    smooth the function curve provided

    :param trajectory: function curve to be smoothed
    :param live: flag indicating whether the function is from video or real-time frames
    :return: smoothed function curve
    """
    new_trajectory = np.copy(trajectory)

    for i in range(3):  # Filter the x, y and angle curves
        if live == 0:
            new_trajectory[:, i] = moving_average(trajectory[:, i], SMOOTHING_RADIUS)
        else:
            new_trajectory[:, i] = moving_average(trajectory[:, i], SMOOTHING_RADIUS+20)
    return new_trajectory


def fix_border(frame):
    """
    fix border for the user-provided frame

    :param frame: user-provided frame
    :return: frame with border fixed
    """
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))
    return frame


def detect_feature(frame):
    """
    get all feature points from the image frame

    :param frame: user-provided frame
    :return: feature points from the image frame
    """
    s = frame.shape
    orb_obj = cv.ORB_create(nfeatures=10000)
    prev_pts = orb_obj.detect(frame, None)
    prev_pts = sorted(prev_pts, key=lambda x: x.response, reverse=True)
    prev_pts = ssc(prev_pts, 1000, 50, s[1], s[0])
    prev_pts = cv.KeyPoint_convert(prev_pts)

    return prev_pts


def transform(transforms_smooth, index, frame, w, h):
    """
    transform the image frame based on the transform information provided

    :param transforms_smooth: smoothed transformation information
    :param index: index of the transforms_smooth for the frame to be processed
    :param frame: frame to be processed
    :param w: width of the frame
    :param h: height of the frame
    :return: smoothed frame
    """
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


def compare_trajectory(original, stabilized):
    """
    produce plots for trajectory comparisons of x, y, and a respectively

    :param original: trajectory for the original video
    :param stabilized: trajectory for the stabilized video
    :return: three plots for trajectory comparisons of x, y, and a respectively
    """
    stabilize_size = len(stabilized)

    plt.figure()
    plt.plot(range(stabilize_size), np.reshape(original[-stabilize_size:, 0], (-1, 1)),
             label='original video')
    plt.plot(range(stabilize_size), np.reshape(stabilized[:, 0], (-1, 1)), label='stabilized video')
    plt.legend(loc="best")
    plt.xlabel('frames')
    plt.ylabel('x variation')
    plt.show()

    plt.figure()
    plt.plot(range(stabilize_size), np.reshape(original[-stabilize_size:, 1], (-1, 1)),
             label='original video')
    plt.plot(range(stabilize_size), np.reshape(stabilized[:, 1], (-1, 1)), label='stabilized video')
    plt.legend(loc="best")
    plt.xlabel('frames')
    plt.ylabel('y variation')
    plt.show()

    plt.figure()
    plt.plot(range(stabilize_size), np.reshape(original[-stabilize_size:, 2], (-1, 1)),
             label='original video')
    plt.plot(range(stabilize_size), np.reshape(stabilized[:, 2], (-1, 1)), label='stabilized video')
    plt.legend(loc="best")
    plt.xlabel('frames')
    plt.ylabel('a variation')
    plt.show()
