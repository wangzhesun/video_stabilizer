# the code is borrowed from the link https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
#   and is modified by Wangzhe Sun
# Import numpy and OpenCV
import numpy as np
import cv2 as cv
from ssc import ssc
import time

# the greater, the smoother, the more likely of black border
SMOOTHING_RADIUS = 30


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    new_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        new_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return new_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    # T = cv.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    T = cv.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.2)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))
    return frame

def stablize(input_video_name, output_video_name, feature_method, test):
    ###### STEP 1 ######
    # Read input video
    cap = cv.VideoCapture(input_video_name)

    # Get frame count
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # fps = cap.get(cv.CAP_PROP_FPS)

    # Define the codec for output video
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    fourcc = cv.VideoWriter_fourcc(*'MJPG')

    # Set up output video
    if output_video_name != 'no_video':
        cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = cv.VideoWriter(output_video_name, fourcc, 20.0, (cols, rows))

    ###### STEP 2 ######
    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

    ###### STEP 3 ######
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    # real time processing
    s = prev_gray.shape

    start = time.time()

    for i in range(n_frames - 2):
        # Fourier Transformation

        # Detect feature points in previous frame
        if feature_method == 'o':
            orb_obj = cv.ORB_create()
            prev_pts = orb_obj.detect(prev_gray, None)
            prev_pts = sorted(prev_pts, key=lambda x: x.response, reverse=True)
            prev_pts = ssc(prev_pts, 1000, 50, s[1], s[0])
            prev_pts = cv.KeyPoint_convert(prev_pts)
        elif feature_method == 's':
            sift_obj = cv.SIFT_create()
            prev_pts = sift_obj.detect(prev_gray, None)
            prev_pts = ssc(prev_pts, 1000, 50, s[1], s[0])
            prev_pts = cv.KeyPoint_convert(prev_pts)
        elif feature_method == 'f':
            fast = cv.FastFeatureDetector_create()
            prev_pts = fast.detect(prev_gray,None)
            prev_pts = cv.KeyPoint_convert(prev_pts)
        else:
            prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                              minDistance=30, blockSize=3)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # The following line has been changed by Wangzhe Sun
        # The original code uses rigid transformation, the new code uses affine transformation
        # Find affine transformation matrix
        # [H, inliers] = cv.estimateRigidTransform(prev_pts, curr_pts)  # will only work with OpenCV-3 or less
        [H, inliers] = cv.estimateAffinePartial2D(prev_pts, curr_pts)
        # print(m)

        # Extract translation
        dx = H[0][2]
        dy = H[1][2]

        # Extract rotation angle
        da = np.arctan2(H[1][0], H[0][0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i + 1) + "/" + str(n_frames - 2) + " -  Tracked points : " + str(
            len(prev_pts)))

    ###### STEP 4 ######
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    if test == 0:
        smoothed_trajectory = smooth(trajectory)

        # Calculate difference in smoothed_trajectory and trajectory
        difference = smoothed_trajectory - trajectory

        # Calculate newer transformation array
        transforms_smooth = transforms + difference

        ###### STEP 5 ######
        # Reset stream to first frame
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        # Write n_frames-1 transformed frames
        for i in range(n_frames - 2):
            # Read next frame
            success, frame = cap.read()
            if not success:
                break

            # Extract transformations from the new transformation array
            dx = transforms_smooth[i, 0]
            dy = transforms_smooth[i, 1]
            da = transforms_smooth[i, 2]

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

            # Fix border artifacts
            frame_stabilized = fixBorder(frame_stabilized)

            if output_video_name != 'no_video':
                # please uncomment the following lines if you want to see side-by-side comparison
                #   between stablized video and original video
                # # Write the frame to the file
                # frame_out = cv.hconcat([frame, frame_stabilized])
                #
                # # If the image is too big, resize it.
                # if frame_out.shape[1] > 1920:
                #     frame_out = cv.resize(frame_out,
                #                           (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)))

                out.write(frame_stabilized)
                # cv.imshow("Before and After", frame_out)

            cv.waitKey(10)

    end = time.time()
    duration = end - start

    if output_video_name != 'no_video':
        out.release()

    return duration, trajectory


if __name__ == '__main__':
    # input_video_name = 'video_1.mp4'
    # input_video_name = 'video_2.mp4'
    input_video_name = 'video_3.mp4'

    print('Stablizing using ORB ...\n')
    [duration_orb, trajectory_orb] = stablize(input_video_name, 'stablized_vid_3_orb.avi', 'o', 0)
    print('Stablizing using GFTT ...\n')
    [duration_gftt, trajectory_gftt] = stablize(input_video_name, 'stablized_vid_3_gftt.avi', 'g', 0)
    print('Stablizing using SIFT ...\n')
    [duration_sift, trajectory_sift] = stablize(input_video_name, 'stablized_vid_3_sift.avi', 's', 0)
    #
    # duration_h = stablize('stablized_vid_orb.avi', 'no_video', 'h')
    print('Testing ORB using FAST ...\n')
    [duration_fast_orb, trajectory_fast_orb] = stablize('stablized_vid_3_orb.avi', 'no_video', 'f', 1)
    print('Testing GFTT using FAST ...\n')
    [duration_fast_gftt, trajectory_fast_gftt] = stablize('stablized_vid_3_gftt.avi', 'no_video', 'f', 1)
    print('Testing SIFT using FAST ...\n')
    [duration_fast_sift, trajectory_fast_sift] = stablize('stablized_vid_3_sift.avi', 'no_video', 'f', 1)
    # trajectory_fast_orb_norm = np.linalg.norm(trajectory_fast_orb / 562) ** 2    # video 1
    # trajectory_fast_gfft_norm = np.linalg.norm(trajectory_fast_gftt / 562) ** 2    # video 1
    # trajectory_fast_sift_norm = np.linalg.norm(trajectory_fast_sift / 562) ** 2      # video 1
    # trajectory_fast_orb_norm = np.linalg.norm(trajectory_fast_orb / 367) ** 2    # video 2
    # trajectory_fast_gfft_norm = np.linalg.norm(trajectory_fast_gftt / 367) ** 2    # video 2
    # trajectory_fast_sift_norm = np.linalg.norm(trajectory_fast_sift / 367) ** 2    # video 2
    trajectory_fast_orb_norm = np.linalg.norm(trajectory_fast_orb / 240) ** 2    # video 3
    trajectory_fast_gfft_norm = np.linalg.norm(trajectory_fast_gftt / 240) ** 2    # video 3
    trajectory_fast_sift_norm = np.linalg.norm(trajectory_fast_sift / 240) ** 2    # video 3
    #
    print('Testing using FAST ...\n')
    [d, t] = stablize(input_video_name, 'no_video', 'f',1)
    # t_tmp = np.linalg.norm(t / 562) ** 2  # video 1
    # t_tmp = np.linalg.norm(t / 367) ** 2  # video 2
    t_tmp = np.linalg.norm(t / 240) ** 2  # video 3
    #
    #
    #
    print('Processing time for using ORB feature point extraction algorithm is ' + str(round(duration_orb,2)) + ' seconds\n')
    print('Processing time for using GFTT feature point extraction algorithm is ' + str(round(duration_gftt, 2)) + ' seconds\n')
    print('Processing time for using SIFT feature point extraction algorithm is ' + str(round(duration_sift, 2)) + ' seconds\n')
    #
    print('Stablization score for using ORB feature point extraction algorithm is ' + str(round(trajectory_fast_orb_norm,2)) + '\n')
    print('Stablization score for using GFTT feature point extraction algorithm is ' + str(round(trajectory_fast_gfft_norm, 2)) + '\n')
    print('Stablization score for using SIFT feature point extraction algorithm is ' + str(round(trajectory_fast_sift_norm, 2)) + '\n')
    print('Stablization score for original is ' + str(round(t_tmp, 2)) + '\n')
