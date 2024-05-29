import cv2 as cv
import face_recognition as fr
import numpy as np
import os
from tkinter import Tk, Button
from tkinter.filedialog import askopenfilename
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Load known encodings
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

# Recognize and label faces in a frame
def recognize_and_label_faces(frame, face_locations, face_encodings):
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.55)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_filenames[first_match_index].split('.')[0]
            expanded_top = max(0, top - 20)
            expanded_bottom = min(frame.shape[0], bottom + 20)
            expanded_left = max(0, left - 20)
            expanded_right = min(frame.shape[1], right + 20)
            cv.rectangle(frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), (0, 255, 0), 2)
            name_width, _ = cv.getTextSize(name, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            cv.rectangle(frame, (expanded_left, expanded_bottom - 20), (expanded_left + name_width + 12, expanded_bottom + 4), (0, 255, 0), cv.FILLED)
            cv.putText(frame, name, (expanded_left + 6, expanded_bottom), cv.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)
        else:
            print("No match found for the current face encoding")

# Process frame for face recognition
def process_frame(frame):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)
    if boxes is not None:
        face_locations = [(int(top), int(right), int(bottom), int(left)) for left, top, right, bottom in boxes]
        face_encodings = fr.face_encodings(rgb_frame, face_locations)
        recognize_and_label_faces(frame, face_locations, face_encodings)
    return frame

# Real-time face recognition
def real_time_recognition():
    video_capture = cv.VideoCapture(0)
    frame_count = 0
    executor = ThreadPoolExecutor(max_workers=2)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 1 == 0:  # Adjust the interval to process every frame
            future = executor.submit(process_frame, frame.copy())
            processed_frame = future.result()
            cv.imshow('Face Recognition', processed_frame)
        else:
            cv.imshow('Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

# Video file face recognition
def video_file_recognition():
    Tk().withdraw()
    video_path = askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
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
        if frame_count % 1 == 0:  # Adjust the interval to process every frame
            future = executor.submit(process_frame, frame.copy())
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

# Main function
def main():
    global known_encodings, known_filenames
    people_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'people'))
    known_encodings, known_filenames = load_known_encodings(people_directory)
    print("Loaded encodings:", known_encodings)
    print("Loaded filenames:", known_filenames)

    root = Tk()
    root.title("Face Recognition")
    Button(root, text="Real-Time Face Recognition", command=real_time_recognition).pack(pady=20)
    Button(root, text="Video File Face Recognition", command=video_file_recognition).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    main()