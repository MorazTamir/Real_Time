# import face_recognition as fr
# import cv2 as cv
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
# import os
# import numpy as np

# #Initialize and open a file dialog to choose an image
# Tk().withdraw()
# load_image = askopenfilename()
# target_image = fr.load_image_file(load_image)
# target_encoding = fr.face_encodings(target_image)
# if not target_encoding:
#     raise ValueError("No faces found in the target image.")

# # Load face encodings from a specified directory
# def preload_encodings(folder):
#     encodings = []
#     filenames = []

#     for filename in os.listdir(folder):
#         filepath = os.path.join(folder, filename)
#         image = fr.load_image_file(filepath)
#         encoding_list = fr.face_encodings(image)
#         if encoding_list:
#             encodings.append(encoding_list[0])
#             filenames.append(filename)

#     return encodings, filenames

# # Create a frame around the face and add a label
# def create_frame(location, label, image):
#     top, right, bottom, left = location
#     cv.rectangle(image, (left, top), (right, bottom), (255,0,0), 2)
#     cv.rectangle(image, (left, bottom + 20), (right, bottom), (255,0,0), cv.FILLED)
#     cv.putText(image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255), 1)

# # Find and mark faces in the image
# def find_target_face(encodings, filenames, image):
#     face_locations = fr.face_locations(image)
#     face_encodings = fr.face_encodings(image, face_locations)

#     for target_enc, location in zip(face_encodings, face_locations):
#         matches = fr.compare_faces(encodings, target_enc, tolerance=0.55)
#         if any(matches):
#             first_match_index = matches.index(True)
#             full_filename = filenames[first_match_index]
#             name, _ = os.path.splitext(full_filename)  # Remove file extension
#             create_frame(location, name, image)

# # Display the image
# def render_image(image):
#     rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     cv.imshow('Face Recognition', rgb_img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# # Implementing the code
# encodings, filenames = preload_encodings('people/')
# editable_image = np.copy(target_image)  # Create a copy of the image for editing
# find_target_face(encodings, filenames, editable_image)
# render_image(editable_image)





# # def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
# #     gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# #     features = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors)
# #     coords=[]
# #     for (x, y, w, h) in features:
# #         cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
# #         cv.putText(img, text, (x, y-4), cv.FONT_HERSHEY_SIMPLEX, 0.8 ,color, 1, cv.LINE_AA)
# #         coords =[x,y,w,h]

# #     return coords,img

# # def detect (img, faceCascade):
# #     color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}

# #     coords,img = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
# #     return img

# # faceCascade = cv.CascadeClassifier('people/')
# # #load camera
# # video = cv.VideoCapture(0)

# # while True:
# #     ret, frame = video.read()
# #     cv.imshow("Face Detection", frame)

# #     key = cv.waitKey(1)
# #     if key == ord('q'):
# #         break

# # video.release()
# # cv.destroyAllWindows()


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
known_encodings, known_filenames = load_known_encodings('people/')

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
