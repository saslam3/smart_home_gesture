import cv2
import os

def frameExtractor(videopath, output_folder, frame_number):
    """
    Extracts a specific frame from a video and saves it as an image.
    """
    cap = cv2.VideoCapture(videopath)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {videopath}. Skipping.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print(f"Error: Failed to extract frame from {videopath}. Skipping.")
        return None  # Return None instead of crashing

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"frame_{frame_number}.jpg")

    cv2.imwrite(filename, frame)
    cap.release()
    
    return filename  # Return the file path if successful
