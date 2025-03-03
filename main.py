import cv2
import numpy as np
import os
import csv
import tensorflow as tf
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

# Ensure Local GPU is enabled
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# ==========================
# Helper Classes
# ==========================
class GestureDetails:
    """ A class to store Gesture details (ID, Name, Label) """

    def __init__(self, gesture_Id, gesture_name, output_label):
        self.gesture_Id = gesture_Id
        self.gesture_name = gesture_name
        self.output_label = output_label


class GestureFeature:
    """ A class to store Gesture feature extraction """

    def __init__(self, gesture_detail: GestureDetails, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature


# ==========================
# Gesture Data Mapping
# ==========================
gesture_data = [
    GestureDetails("Num0", "0", "0"), GestureDetails("Num1", "1", "1"),
    GestureDetails("Num2", "2", "2"), GestureDetails("Num3", "3", "3"),
    GestureDetails("Num4", "4", "4"), GestureDetails("Num5", "5", "5"),
    GestureDetails("Num6", "6", "6"), GestureDetails("Num7", "7", "7"),
    GestureDetails("Num8", "8", "8"), GestureDetails("Num9", "9", "9"),
    GestureDetails("FanDown", "Decrease Fan Speed", "10"),
    GestureDetails("FanOn", "FanOn", "11"), GestureDetails("FanOff", "FanOff", "12"),
    GestureDetails("FanUp", "Increase Fan Speed", "13"),
    GestureDetails("LightOff", "LightOff", "14"), GestureDetails("LightOn", "LightOn", "15"),
    GestureDetails("SetThermo", "SetThermo", "16")
]

# ==========================
# Functions
# ==========================
def extract_feature(folder_path, input_file, mid_frame_counter):
    """ Extract features from the middle frame of a video """
    middle_image = cv2.imread(frameExtractor(folder_path + input_file, folder_path + "frames/", mid_frame_counter),
                              cv2.IMREAD_GRAYSCALE)
    feature_extracted = HandShapeFeatureExtractor.extract_feature(
        HandShapeFeatureExtractor.get_instance(), middle_image)
    return feature_extracted


def get_gesture_by_file_name(gesture_file_name):
    """ Find gesture details based on the file name """
    for gesture in gesture_data:
        if gesture.gesture_Id == gesture_file_name.split('_')[0]:
            return gesture
    return None


# ==========================
# Training Feature Extraction
# ==========================
featureVectorList = []
train_data_path = "traindata/"

# Ensure training folder exists
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Error: Training data folder '{train_data_path}' is missing.")

for count, file in enumerate(os.listdir(train_data_path)):
    if not file.startswith('frames'):
        gesture_detail = get_gesture_by_file_name(file)
        extracted_feature = extract_feature(train_data_path, file, count)

        if extracted_feature is not None:
            featureVectorList.append(GestureFeature(gesture_detail, extracted_feature))
        else:
            print(f"Warning: Feature extraction failed for {file}")

# ==========================
# Gesture Recognition
# ==========================
def gesture_detection(gesture_folder_path, gesture_file_name, mid_frame_counter):
    """ Recognize gesture using cosine similarity """
    video_feature = extract_feature(gesture_folder_path, gesture_file_name, mid_frame_counter)

    if video_feature is None:
        print(f"Error: Feature extraction failed for {gesture_file_name}")
        return None

    similarity = 1
    best_match = None
    for featureVector in featureVectorList:
        cosine_similarity = tf.keras.losses.cosine_similarity(video_feature, featureVector.extracted_feature, axis=-1)
        if cosine_similarity < similarity:
            similarity = cosine_similarity
            best_match = featureVector.gesture_detail

    return best_match

# ==========================
# Test Feature Extraction & Results
# ==========================
test_data_path = "test/"
results_file_path = os.path.join(os.path.dirname(__file__), "results.csv")

# Ensure test folder exists
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Error: Test data folder '{test_data_path}' is missing.")

test_files = [file for file in os.listdir(test_data_path) if not file.startswith('frames')]
if not test_files:
    raise FileNotFoundError("Error: No test videos found in the 'test/' folder.")

# Generate results.csv
with open(results_file_path, 'w', newline='') as results_file:
    fields_names = ['Gesture_Video_File_Name', 'Gesture_Name', 'Output_Label']
    data_writer = csv.DictWriter(results_file, fieldnames=fields_names)
    data_writer.writeheader()

    for test_count, test_file in enumerate(test_files):
        recognized_gesture_detail = gesture_detection(test_data_path, test_file, test_count)

        if recognized_gesture_detail is None:
            print(f"Warning: No gesture detected for {test_file}")
            recognized_gesture_detail = GestureDetails("Unknown", "Unknown Gesture", "-1")

        data_writer.writerow({
            'Gesture_Video_File_Name': test_file,
            'Gesture_Name': recognized_gesture_detail.gesture_name,
            'Output_Label': recognized_gesture_detail.output_label
        })

print(f"✅ Results saved in {results_file_path}")
