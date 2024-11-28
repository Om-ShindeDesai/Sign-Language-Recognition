import cv2
import mediapipe as mp
from model import KeyPointClassifier
import landmark_utils as u
import time
import numpy as np  # Import numpy for array manipulation
import copy
import itertools
from collections import deque


kpclf = KeyPointClassifier()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic  # Holistic model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                               )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                               )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                               )

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    if not landmark_list:
        return []  # Return an empty list if the input is empty

    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    if not temp_landmark_list:
        return []  # Return an empty list if the list is empty after flattening

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def extract_keypoints(results, image):
    pose_landmarks = calc_landmark_list(image, results.pose_landmarks) if results.pose_landmarks else []
    lh_landmarks = calc_landmark_list(image, results.left_hand_landmarks) if results.left_hand_landmarks else []
    rh_landmarks = calc_landmark_list(image, results.right_hand_landmarks) if results.right_hand_landmarks else []

    pose_landmarks = pre_process_landmark(pose_landmarks)
    lh_landmarks = pre_process_landmark(lh_landmarks)
    rh_landmarks = pre_process_landmark(rh_landmarks)

    # Create fixed-length arrays
    n_pose_landmarks = 33 * 2
    n_hand_landmarks = 21 * 2
    pose_landmarks = np.pad(pose_landmarks, (0, n_pose_landmarks - len(pose_landmarks)), mode='constant')
    lh_landmarks = np.pad(lh_landmarks, (0, n_hand_landmarks - len(lh_landmarks)), mode='constant')
    rh_landmarks = np.pad(rh_landmarks, (0, n_hand_landmarks - len(rh_landmarks)), mode='constant')

    keypoints = np.concatenate([pose_landmarks, lh_landmarks, rh_landmarks])
    return keypoints


gestures = {
    0: "None",
    1: "Hello",
    2: "Woman",
    3: "Man",
    4: "Peace",
    5: "Victory",
    6: "Good",
    7: "India",
    8: "Welcome",
    9: "Sorry ",
    10: "Time",
    11: "Bad",
    12: "Difficult ",
    13: "Easy ",
    14: "Strong ",
    15: "What",
    16: "You",
    17: "Me",
    18: "Camera",
    19: "Flat",
    20: "Big",
    21: "Small",
    22: "Marry",
    23: "Place",
    24: "Teacher",
    25: "Happy",
    26: "Question",
    27: "Answer",
    28: "Namaste",
    29: "House",
    30: "Fat",
    31: "Morning",
    32: "Afternoon"
}


def main():
    keypoints = None
    cap = cv2.VideoCapture(0)
    gesture_counter = {}
    current_gesture = None
    gesture_threshold = 5
    recent_gestures = deque(maxlen=5)
    last_added_gesture = None
    gesture_sequences = {
        (1, 2): "Hello, woman!",
        (4, 3): "Peace, man!",
        # Add more gesture sequences and their corresponding sentences here
    }
    recognition_active = False  # Flag to indicate whether recognition is active
    space_pressed = False  # Flag to track spacebar press
    sequence_timer_start = None  # Timer to track sequence display duration
    sequence_display_duration = 1.0  # Duration to display sequence in seconds

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_pose_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            current_time = time.time()
            if current_time - last_pose_time >= 0.2:
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results, image)
                gesture_index = kpclf(keypoints)

                if recognition_active:  # Check if recognition is active
                    gesture_counter[gesture_index] = gesture_counter.get(gesture_index, 0) + 1
                    if gesture_counter[gesture_index] >= gesture_threshold:
                        current_gesture = gesture_index
                        if gesture_index != last_added_gesture:
                            if gesture_index == 0:
                                recent_gestures.clear()  # Empty the array if gesture_index is 0
                                sequence_timer_start = None  # Reset the sequence timer
                            else:
                                recent_gestures.append(gesture_index)
                                last_added_gesture = gesture_index
                                sequence_timer_start = time.time()  # Start the sequence timer

                    else:
                        current_gesture = None

                    for idx, count in gesture_counter.items():
                        if idx != gesture_index:
                            gesture_counter[idx] = 0

                    sequence = tuple(recent_gestures)
                    print(sequence)
                    # Check for longer sequences first
                    for i in range(len(sequence), 0, -1):
                        sub_sequence = sequence[-i:]
                        if sub_sequence in gesture_sequences:
                            sentence = gesture_sequences[sub_sequence]
                            cv2.putText(image, sentence, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
                            if sequence_timer_start is not None:
                                if time.time() - sequence_timer_start >= sequence_display_duration:
                                    recent_gestures.clear()  # Empty the queue after display duration
                                    sequence_timer_start = None  # Reset the sequence timer
                            break  # Break the loop once a sub-sequence is found

                    if current_gesture is not None and sequence not in gesture_sequences:
                        gesture_text = gestures[current_gesture]
                        cv2.putText(image, gesture_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                final_image = cv2.flip(image, 1)
                cv2.imshow('MediaPipe Hands', final_image)

            key = cv2.waitKey(5)
            if key == 27:  # ESC key to exit
                break
            elif key == 32:  # Spacebar key to toggle recognition
                space_pressed = not space_pressed
                recognition_active = space_pressed  # Set recognition active based on spacebar press

        cap.release()
        cv2.destroyAllWindows()

if _name_ == '_main_':
    main()
