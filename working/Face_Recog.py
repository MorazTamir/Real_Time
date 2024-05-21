import cv2 as cv
import face_recognition as fr
import numpy as np
import os
from tkinter import Tk, Button
from tkinter.filedialog import askopenfilename

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

# Initialize known face encodings
people_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'people'))
known_encodings, known_filenames = load_known_encodings(people_directory)

# Real-time face recognition
def real_time_recognition():
    video_capture = cv.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.55)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_filenames[first_match_index].split('.')[0]
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv.putText(frame, name, (left + 6, bottom - 6), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

        cv.imshow('Face Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

# Face recognition on existing video
def video_file_recognition():
    Tk().withdraw()
    video_path = askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        return

    video_capture = cv.VideoCapture(video_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.55)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_filenames[first_match_index].split('.')[0]
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv.putText(frame, name, (left + 6, bottom - 6), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

        cv.imshow('Face Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

# Main GUI
def main():
    root = Tk()
    root.title("Face Recognition")
    
    Button(root, text="Real-Time Face Recognition", command=real_time_recognition).pack(pady=20)
    Button(root, text="Video File Face Recognition", command=video_file_recognition).pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    main()
