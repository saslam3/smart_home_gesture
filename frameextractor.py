
import cv2
import os

"""
videopath : path of the video file
frames_path: path of the directory to which the frames are saved
count: to assign the video order to the frame.
"""


def frameExtractor(videopath, frames_path, count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no = int(video_length / 2)
    # print("Extracting frame..\n")
    cap.set(1, frame_no)
    ret, frame = cap.read()
    # cv2.imwrite(frames_path + "/%#05d.png" % (count+1), frame)
    filename = frames_path + "%#05d.png" % (count + 1)
    if frame is None or frame.size == 0:
        print(f"Error: Failed to extract frame from {video_path}. Skipping.")
        return None  # Return None instead of crashing

    cv2.imwrite(filename, frame)
    return filename
