import cv2
import csv
import mediapipe as mp
import numpy as np
import copy
import itertools
import time

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

def log_csv(sign_name, keypoints):
    if sign_name is None:
        return

    csv_path = 'model/keypoint_classifier/keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        sign_name_array = np.array([sign_name], dtype=np.int32)  # Convert sign_name to a NumPy array of integers
        combined_row = np.concatenate([sign_name_array, keypoints])
        writer.writerow(combined_row)

def main():
    cap = cv2.VideoCapture(0)
    sign_name = None
    keypoints = None  # Initialize keypoints to None
    capture_keypoints = False  # Flag to control keypoints capturing

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            received_key = cv2.waitKey(20)

            if received_key == 32:  # Spacebar key
                if sign_name is None:
                    sign_name = int(input())
                    capture_keypoints = True  # Set the flag to start capturing keypoints after 2 seconds
                    time.sleep(2)  # Wait for 2 seconds
                else:
                    sign_name = None  # Reset sign_name
                    keypoints = None  # Reset keypoints
                    capture_keypoints = False  # Reset the flag

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            if capture_keypoints and sign_name is not None:
                keypoints = extract_keypoints(results, image)
                log_csv(sign_name, keypoints)  # Log keypoints

            text = f"Sign Name: {sign_name}" if sign_name is not None else "Press spacebar to enter/change sign name"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

            cv2.imshow('MediaPipe Hands and Pose', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

if __name__ == '__main__':
    main()