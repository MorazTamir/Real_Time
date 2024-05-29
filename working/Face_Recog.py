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
people_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'people')) # type: ignore
known_encodings, known_filenames = load_known_encodings(people_directory)
print("Loaded encodings:", known_encodings)
print("Loaded filenames:", known_filenames)

def recognize_and_label_faces(frame, face_locations, face_encodings):
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.55)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_filenames[first_match_index].split('.')[0]
            # Expand the bounding box to include the entire face
            expanded_top = max(0, top - 20)
            expanded_bottom = min(frame.shape[0], bottom + 20)
            expanded_left = max(0, left - 20)
            expanded_right = min(frame.shape[1], right + 20)
            cv.rectangle(frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), (0, 255, 0), 2)
            # Add a background behind the name
            name_width, _ = cv.getTextSize(name, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            cv.rectangle(frame, (expanded_left, expanded_bottom - 20), (expanded_left + name_width + 12, expanded_bottom + 4), (0, 255, 0), cv.FILLED)
            # Add the name on top of the background
            cv.putText(frame, name, (expanded_left + 6, expanded_bottom), cv.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)
        else:
            print("No match found for the current face encoding")

def real_time_recognition():
    video_capture = cv.VideoCapture(0)
    process_frame_interval = 1  # Adjust the interval to 1 to process every frame
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_frame_interval == 0:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame, face_locations)
            print("Face locations:", face_locations)
            print("Face encodings:", face_encodings)
            recognize_and_label_faces(frame, face_locations, face_encodings)

        cv.imshow('Face Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

def video_file_recognition():
    Tk().withdraw()
    video_path = askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
    if not video_path:
        return
    video_capture = cv.VideoCapture(video_path)
    process_frame_interval = 1  # Adjust the interval to 1 to process every frame
    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_frame_interval == 0:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame, face_locations)
            print("Face locations:", face_locations)
            print("Face encodings:", face_encodings)
            recognize_and_label_faces(frame, face_locations, face_encodings)

        cv.imshow('Face Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Skip 30 frames ahead
        for _ in range(30):
            video_capture.grab()

    video_capture.release()
    cv.destroyAllWindows()

def main():
    root = Tk()
    root.title("Face Recognition")
    Button(root, text="Real-Time Face Recognition", command=real_time_recognition).pack(pady=20)
    Button(root, text="Video File Face Recognition", command=video_file_recognition).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    main()