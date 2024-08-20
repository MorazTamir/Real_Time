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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Dictionary to keep track of the last time each person was greeted
last_greeted = {}

def speak_greeting(name):
    engine.say(f"Hello {name}")
    engine.runAndWait()

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

def recognize_and_label_faces(frame, face_locations, face_encodings, known_encodings, known_filenames):
    recognized_names = []
    current_time = time.time()
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.55)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_filenames[first_match_index].split('.')[0]
            recognized_names.append(name)
            expanded_top = max(0, top - 20)
            expanded_bottom = min(frame.shape[0], bottom + 20)
            expanded_left = max(0, left - 20)
            expanded_right = min(frame.shape[1], right + 20)
            cv.rectangle(frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), (0, 255, 0), 2)
            name_width, _ = cv.getTextSize(name, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            cv.rectangle(frame, (expanded_left, expanded_bottom - 20), (expanded_left + name_width + 12, expanded_bottom + 4), (0, 255, 0), cv.FILLED)
            cv.putText(frame, name, (expanded_left + 6, expanded_bottom), cv.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)

            if name not in last_greeted or current_time - last_greeted[name] > 60:
                Thread(target=speak_greeting, args=(name,)).start()
                last_greeted[name] = current_time

    return recognized_names

def process_frame(frame, known_encodings, known_filenames):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)
    if boxes is not None:
        face_locations = [(int(top), int(right), int(bottom), int(left)) for left, top, right, bottom in boxes]
        face_encodings = fr.face_encodings(rgb_frame, face_locations)
        recognize_and_label_faces(frame, face_locations, face_encodings, known_encodings, known_filenames)
    return frame

def real_time_recognition(known_encodings, known_filenames):
    video_capture = cv.VideoCapture(0)
    frame_count = 0
    executor = ThreadPoolExecutor(max_workers=2)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 1 == 0:
            future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
            processed_frame = future.result()
            cv.imshow('Face Recognition', processed_frame)
        else:
            cv.imshow('Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

def video_file_recognition(known_encodings, known_filenames):
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
    if not video_path:
        return

    video_capture = cv.VideoCapture(video_path)
    frame_count = 0
    executor = ThreadPoolExecutor(max_workers=2)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 1 == 0:
            future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
            processed_frame = future.result()
            cv.imshow('Face Recognition', processed_frame)
        else:
            cv.imshow('Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        for _ in range(30):  # Skip 30 frames ahead
            video_capture.grab()

    video_capture.release()
    cv.destroyAllWindows()

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
