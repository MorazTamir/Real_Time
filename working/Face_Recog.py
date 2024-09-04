import os
import cv2 as cv
import face_recognition as fr
import numpy as np
import shutil
from tkinter import filedialog, simpledialog
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import time
import pyttsx3
from facenet_pytorch import MTCNN
import torch

# Constants
TOLERANCE = 0.55
FRAME_SKIP = 30
MAX_WORKERS = 2
VIDEO_SOURCE_INDEX = 0
GREETING_INTERVAL = 60  # in seconds
NAME_FONT_SCALE = 0.4
NAME_FONT_THICKNESS = 1
RECTANGLE_COLOR = (0, 255, 0)
RECTANGLE_THICKNESS = 2
WINDOW_NAME = 'Face Recognition'
FACE_PADDING = 20  # Padding around the face in pixels
TEXT_COLOR = (0, 0, 0)  # Black color for text
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Dictionary to keep track of the last time each person was greeted
last_greeted = {}

# Function to speak a greeting asynchronously
def speak_greeting(name):
    engine.say(f"Hello {name}")
    engine.runAndWait()

# Function to load face encodings from images in a specified folder
def load_known_encodings(folder):
    encodings = []
    filenames = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        image = fr.load_image_file(filepath)
        encoding_list = fr.face_encodings(image)
        if encoding_list:
            encodings.append(encoding_list[0])
            filenames.append(filename)
    return encodings, filenames

# Function to detect the color of the eyes
def detect_eye_color(eye_region):
    # Convert the eye region to the HSV color space
    hsv = cv.cvtColor(eye_region, cv.COLOR_BGR2HSV)

    # Define color ranges for blue, green, and brown eyes in HSV
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])

    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

    brown_lower = np.array([10, 20, 20])
    brown_upper = np.array([20, 255, 255])

    # Create masks for each color
    blue_mask = cv.inRange(hsv, blue_lower, blue_upper)
    green_mask = cv.inRange(hsv, green_lower, green_upper)
    brown_mask = cv.inRange(hsv, brown_lower, brown_upper)

    # Calculate the percentage of each color in the eye region
    blue_percentage = np.sum(blue_mask) / (eye_region.size)
    green_percentage = np.sum(green_mask) / (eye_region.size)
    brown_percentage = np.sum(brown_mask) / (eye_region.size)

    # Determine the most dominant color
    if blue_percentage > green_percentage and blue_percentage > brown_percentage:
        return "Blue"
    elif green_percentage > blue_percentage and green_percentage > brown_percentage:
        return "Green"
    else:
        return "Brown"

# Function to recognize and label faces in a given frame
def recognize_and_label_faces(frame, face_locations, face_encodings, known_encodings, known_filenames):
    recognized_names = []
    current_time = time.time()
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
        name = "Unknown"
        eye_color = None
        if True in matches:
            first_match_index = matches.index(True)
            name = known_filenames[first_match_index].split('.')[0]
            recognized_names.append(name)
            expanded_top = max(0, top - FACE_PADDING)
            expanded_bottom = min(frame.shape[0], bottom + FACE_PADDING)
            expanded_left = max(0, left - FACE_PADDING)
            expanded_right = min(frame.shape[1], right + FACE_PADDING)
            cv.rectangle(frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), RECTANGLE_COLOR, RECTANGLE_THICKNESS)
            name_width, _ = cv.getTextSize(name, cv.FONT_HERSHEY_DUPLEX, NAME_FONT_SCALE, NAME_FONT_THICKNESS)[0]
            cv.rectangle(frame, (expanded_left, expanded_bottom - 20), (expanded_left + name_width + 12, expanded_bottom + 40), RECTANGLE_COLOR, cv.FILLED)
            cv.putText(frame, name, (expanded_left + 6, expanded_bottom), cv.FONT_HERSHEY_DUPLEX, NAME_FONT_SCALE, TEXT_COLOR, NAME_FONT_THICKNESS)

            # Detect eyes and determine eye color
            eyes = mtcnn.detect(frame[expanded_top:expanded_bottom, expanded_left:expanded_right])
            if eyes[0] is not None:
                for (eye_left, eye_top, eye_right, eye_bottom) in eyes[0]:
                    eye_region = frame[int(eye_top):int(eye_bottom), int(eye_left):int(eye_right)]
                    eye_color = detect_eye_color(eye_region)
                    cv.putText(frame, f"Eyes: {eye_color}", (expanded_left + 6, expanded_bottom + 15), cv.FONT_HERSHEY_DUPLEX, NAME_FONT_SCALE, TEXT_COLOR, NAME_FONT_THICKNESS)

            if name not in last_greeted or current_time - last_greeted[name] > GREETING_INTERVAL:
                Thread(target=speak_greeting, args=(name,)).start()
                last_greeted[name] = current_time

    return recognized_names

# Unified function to process a video frame for face recognition
def process_frame(frame, known_encodings, known_filenames):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)
    if boxes is not None:
        face_locations = [(int(top), int(right), int(bottom), int(left)) for left, top, right, bottom in boxes]
        face_encodings = fr.face_encodings(rgb_frame, face_locations)
        recognize_and_label_faces(frame, face_locations, face_encodings, known_encodings, known_filenames)
    return frame

# Function to recognize faces in real-time from webcam feed
def real_time_recognition(known_encodings, known_filenames):
    video_capture = cv.VideoCapture(VIDEO_SOURCE_INDEX)
    frame_count = 0
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 1 == 0:
            future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
            processed_frame = future.result()
            cv.imshow(WINDOW_NAME, processed_frame)
        else:
            cv.imshow(WINDOW_NAME, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

# Function to recognize faces in a video file
def video_file_recognition(known_encodings, known_filenames):
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
    if not video_path:
        return

    video_capture = cv.VideoCapture(video_path)
    frame_count = 0
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 1 == 0:
            future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
            processed_frame = future.result()
            cv.imshow(WINDOW_NAME, processed_frame)
        else:
            cv.imshow(WINDOW_NAME, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        for _ in range(FRAME_SKIP):  # Skip frames
            video_capture.grab()

    video_capture.release()
    cv.destroyAllWindows()

# Function to add new photos to the people directory and update known encodings
def add_photos(people_directory):
    files = filedialog.askopenfilenames(initialdir=os.path.expanduser('~\\Downloads'), filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if files:
        for file in files:
            name = simpledialog.askstring("Input", f"Enter name for {os.path.basename(file)}:")
            if name:
                new_filename = os.path.join(people_directory, f"{name}.jpg")
                shutil.copy(file, new_filename)
                print(f"Added {new_filename}")

        # Reload encodings after adding new photos
        known_encodings, known_filenames = load_known_encodings(people_directory)
        print("Reloaded encodings with new photos")
