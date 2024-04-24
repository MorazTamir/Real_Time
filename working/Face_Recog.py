import cv2 as cv
import face_recognition as fr
import numpy as np
import os

# Load known face encodings and their corresponding filenames
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

# Load the known face encodings
# known_encodings, known_filenames = load_known_encodings('../people/')
# Get the absolute path to the 'people' directory
people_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'people'))
# Load the known face encodings
known_encodings, known_filenames = load_known_encodings(people_directory)

# Initialize video capture from webcam
video_capture = cv.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face encoding with the known face encodings
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.55)

        # If a match is found, mark the face with a rectangle and label it
        if True in matches:
            first_match_index = matches.index(True)
            name = known_filenames[first_match_index]
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(frame, name, (left + 6, bottom - 6), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv.destroyAllWindows()
