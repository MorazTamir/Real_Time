import cv2 as cv
import face_recognition as fr
import numpy as np
import os
from tkinter import Tk, Button, Canvas, filedialog, simpledialog
from tkinter.filedialog import askopenfilenames
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageTk
import shutil
import pyttsx3
import time
from threading import Thread

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Global variable for people directory
people_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'people'))

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Dictionary to keep track of the last time each person was greeted
last_greeted = {}

# Function to speak a greeting asynchronously
def speak_greeting(name):
    engine.say(f"Hello {name}")
    engine.runAndWait()

# Skin Color Extraction Function
def extract_skin_color(image):
    # Convert image to HSV color space
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    skin_mask = cv.inRange(hsv_image, lower_skin, upper_skin)
    
    return skin_mask

# Feature Extraction Function
def extract_skin_color_features(image, mask):
    # Calculate the average color of the skin region
    skin_color = cv.mean(image, mask=mask)[:3]
    return skin_color

# Matching Algorithm Function
def match_skin_color(input_features, dataset_features):
    # Implement your matching algorithm here
    closest_matches = None
    return closest_matches

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
            
            # Greet the person if they haven't been greeted recently
            if name not in last_greeted or current_time - last_greeted[name] > 60:
                Thread(target=speak_greeting, args=(name,)).start()
                last_greeted[name] = current_time
    
    return recognized_names

# Process frame for face recognition
def process_frame(frame, known_encodings, known_filenames):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)
    if boxes is not None:
        face_locations = [(int(top), int(right), int(bottom), int(left)) for left, top, right, bottom in boxes]
        face_encodings = fr.face_encodings(rgb_frame, face_locations)
        recognized_names = recognize_and_label_faces(frame, face_locations, face_encodings, known_encodings, known_filenames)
        if recognized_names:
            print("Recognized names:", recognized_names)
    return frame

# Real-time face recognition
def real_time_recognition():
    video_capture = cv.VideoCapture(0)
    frame_count = 0
    executor = ThreadPoolExecutor(max_workers=2)
    global known_encodings, known_filenames

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 1 == 0:  # Adjust the interval to process every frame
            future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
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
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
    if not video_path:
        return

    video_capture = cv.VideoCapture(video_path)
    frame_count = 0
    executor = ThreadPoolExecutor(max_workers=2)
    global known_encodings, known_filenames

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 1 == 0:  # Adjust the interval to process every frame
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

# Add new photos to the folder
def add_photos():
    global people_directory
    files = filedialog.askopenfilenames(initialdir=os.path.expanduser('~\\Downloads'), filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if files:
        for file in files:
            name = simpledialog.askstring("Input", f"Enter name for {os.path.basename(file)}:")
            if name:
                # Copy the file to the people_directory folder with the new name
                new_filename = os.path.join(people_directory, f"{name}.jpg")
                shutil.copy(file, new_filename)
                print(f"Added {new_filename}")
        
        # Reload encodings after adding new photos
        global known_encodings, known_filenames
        known_encodings, known_filenames = load_known_encodings(people_directory)
        print("Reloaded encodings with new photos")

# Main function
def main():
    global known_encodings, known_filenames, people_directory
    known_encodings, known_filenames = load_known_encodings(people_directory)
    print("Loaded encodings:", known_encodings)
    print("Loaded filenames:", known_filenames)

    root = Tk()
    root.title("Face Recognition")
    root.geometry("300x300")  # Larger size for better display
    root.resizable(False, False)

    # Load and display background image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bg_image_path = os.path.join(script_dir, "background.jpg")
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((300, 300), Image.Resampling.LANCZOS)  # Adjust size
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = Canvas(root, width=300, height=300)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Add buttons on top of the background
    btn_real_time = Button(root, text="Real-Time Face Recognition", command=real_time_recognition)
    btn_video_file = Button(root, text="Video File Face Recognition", command=video_file_recognition)
    btn_add_photos = Button(root, text="Add Photos", command=add_photos)

    # Place buttons on the canvas
    canvas.create_window(150, 125, window=btn_real_time)  # Adjust positions as needed
    canvas.create_window(150, 175, window=btn_video_file)
    canvas.create_window(150, 225, window=btn_add_photos)

    root.mainloop()

if __name__ == "__main__":
    main()
