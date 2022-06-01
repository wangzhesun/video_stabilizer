# the code is borrowed from the link https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
#   and is modified by Wangzhe Sun
# Import numpy and OpenCV
import numpy as np
import cv2 as cv
from utils import util
import time


class Stabilizer:
    def __init__(self):
        self.mode_ = 'live'
        self.output_ = 0

    def set_mode(self, mode, output=0):
        self.mode_ = mode
        self.output_ = output

    def stabilize_video(self, input_vid, output_vid):
        cap = cv.VideoCapture(input_vid)  # Read input video
        assert cap.isOpened(), 'Cannot capture source'

        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Get frame count
        # Get width and height of video stream
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Set up output video
        if self.output_:
            cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Define the codec for output video
            out = cv.VideoWriter(output_vid, fourcc, 20.0, (cols, rows))

        _, prev = cap.read()  # Read first frame
        prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

        # Pre-define transformation-store array
        transforms = np.zeros((n_frames - 1, 3), np.float32)
        start = time.time()

        for i in range(n_frames - 2):
            # Detect feature points in previous frame
            prev_pts = util.detect_feature(prev_gray)

            success, curr = cap.read()  # Read next frame
            if not success:
                break

            curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            assert prev_pts.shape == curr_pts.shape

            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Find affine transformation matrix
            [H, _] = cv.estimateAffinePartial2D(prev_pts, curr_pts)

            # Extract translation
            dx = H[0][2]
            dy = H[1][2]
            da = np.arctan2(H[1][0], H[0][0])  # Extract rotation angle
            transforms[i] = [dx, dy, da]  # Store transformation

            prev_gray = curr_gray  # Move to next frame

            print("Frame: " + str(i + 1) + "/" + str(
                n_frames - 2) + " -  Tracked points : " + str(
                len(prev_pts)))

        # Compute trajectory using cumulative sum of transformations
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = util.smooth(trajectory)
        # Calculate difference in smoothed_trajectory and trajectory
        difference = smoothed_trajectory - trajectory
        # Calculate newer transformation array
        transforms_smooth = transforms + difference

        cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Reset stream to first frame

        # Write n_frames-1 transformed frames
        for i in range(n_frames - 2):
            # Read next frame
            success, frame = cap.read()
            if not success:
                break

            frame_stabilized = util.transform(transforms_smooth, i, frame, w, h)
            if self.output_:
                out.write(frame_stabilized)

            frame_out = cv.hconcat([frame, frame_stabilized])
            if frame_out.shape[1] > 1920:
                frame_out = cv.resize(frame_out,
                                      (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)))
            cv.imshow("Before and After", frame_out)

            cv.waitKey(10)

        end = time.time()
        duration = end - start
        print("Video stabilized in {} seconds\n".format(round(duration, 2)))

        if self.output_:
            out.release()

    def stabilize_live(self, output_vid):
        cap = cv.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source'

        # Get width and height of video stream
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Set up output video
        if self.output_:
            cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Define the codec for output video
            out = cv.VideoWriter(output_vid, fourcc, 20.0, (cols, rows))

        # Pre-define transformation-store array
        # transforms = np.zeros((n_frames - 1, 3), np.float32)
        transforms = np.zeros((1, 3), np.float32)
        trajectory = np.zeros((1, 3), np.float32)

        frames = 0

        ret, prev = cap.read()  # Read first frame

        while cap.isOpened():
            prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
            prev_pts = util.detect_feature(prev_gray)

            success, curr = cap.read()  # Read next frame
            if not success:
                break

            curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            assert prev_pts.shape == curr_pts.shape

            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Find affine transformation matrix
            [H, _] = cv.estimateAffinePartial2D(prev_pts, curr_pts)

            # Extract translation
            dx = H[0][2]
            dy = H[1][2]
            da = np.arctan2(H[1][0], H[0][0])  # Extract rotation angle
            transforms = np.append(transforms, [[dx, dy, da]], axis=0)

            # transforms[i] = [dx, dy, da]  # Store transformation

            prev_gray = curr_gray  # Move to next frame

            print("Frame: " + str(frames + 1) + " -  Tracked points : " + str(len(prev_pts)))

            trajectory = np.append(trajectory, [trajectory[-1, :] + transforms[-1, :]], axis=0)
            # trajectory = np.append(trajectory, [np.cumsum(transforms[1:,:], axis=0)])

            smoothed_trajectory = util.smooth(trajectory[1:, :])
            # Calculate difference in smoothed_trajectory and trajectory
            difference = smoothed_trajectory - trajectory[1:, :]
            # Calculate newer transformation array
            transforms_smooth = transforms[1:, :] + difference

            frame_stabilized = util.transform(transforms_smooth, -1, curr, w, h)
            if self.output_:
                out.write(frame_stabilized)

            frame_out = cv.hconcat([curr, frame_stabilized])
            if frame_out.shape[1] > 1920:
                frame_out = cv.resize(frame_out,
                                      (int(frame_out.shape[1] / 2),
                                       int(frame_out.shape[0] / 2)))
            cv.imshow("Before and After", frame_out)

            cv.waitKey(10)

            prev = curr

        if self.output_:
            out.release()

            # for i in range(n_frames - 2):
            #     # Detect feature points in previous frame
            #     prev_pts = util.detect_feature(prev_gray)
            #
            #     success, curr = cap.read()  # Read next frame
            #     if not success:
            #         break
            #
            #     curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
            #
            #     # Calculate optical flow (i.e. track feature points)
            #     curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts,
            #                                                     None)
            #     assert prev_pts.shape == curr_pts.shape
            #
            #     # Filter only valid points
            #     idx = np.where(status == 1)[0]
            #     prev_pts = prev_pts[idx]
            #     curr_pts = curr_pts[idx]
            #
            #     # Find affine transformation matrix
            #     [H, _] = cv.estimateAffinePartial2D(prev_pts, curr_pts)
            #
            #     # Extract translation
            #     dx = H[0][2]
            #     dy = H[1][2]
            #     da = np.arctan2(H[1][0], H[0][0])  # Extract rotation angle
            #     transforms[i] = [dx, dy, da]  # Store transformation
            #
            #     prev_gray = curr_gray  # Move to next frame
            #
            #     print("Frame: " + str(i + 1) + "/" + str(
            #         n_frames - 2) + " -  Tracked points : " + str(
            #         len(prev_pts)))

            # Compute trajectory using cumulative sum of transformations
            # trajectory = np.cumsum(transforms, axis=0)
            # smoothed_trajectory = util.smooth(trajectory)
            # # Calculate difference in smoothed_trajectory and trajectory
            # difference = smoothed_trajectory - trajectory
            # # Calculate newer transformation array
            # transforms_smooth = transforms + difference
            #
            # cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Reset stream to first frame
            #
            # # Write n_frames-1 transformed frames
            # for i in range(n_frames - 2):
            #     # Read next frame
            #     success, frame = cap.read()
            #     if not success:
            #         break
            #
            #     frame_stabilized = util.transform(transforms_smooth, i, frame, w, h)
            #     if self.output_:
            #         out.write(frame_stabilized)
            #
            #     frame_out = cv.hconcat([frame, frame_stabilized])
            #     if frame_out.shape[1] > 1920:
            #         frame_out = cv.resize(frame_out,
            #                               (int(frame_out.shape[1] / 2),
            #                                int(frame_out.shape[0] / 2)))
            #     cv.imshow("Before and After", frame_out)
            #
            #     cv.waitKey(10)
            #
            # end = time.time()
            # duration = end - start
            # print("Video stabilized in {} seconds\n".format(round(duration, 2)))
            #
            # if self.output_:
            #     out.release()

    def stabilize(self, input_vid="", output_vid=""):
        if self.mode_ == 'video':
            self.stabilize_video(input_vid, output_vid)
        else:
            self.stabilize_live(output_vid)


if __name__ == '__main__':
    stabilizer = Stabilizer()
    stabilizer.set_mode('live', 0)
    stabilizer.stabilize()
