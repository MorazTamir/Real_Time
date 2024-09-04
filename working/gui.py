import os
from tkinter import Tk, Button, Canvas, filedialog, simpledialog
from PIL import Image, ImageTk
from Face_Recog import real_time_recognition, video_file_recognition, add_photos

# Constants
WINDOW_TITLE = "Face Recognition"  # Title of the GUI window
WINDOW_SIZE = "300x300"  # Size of the GUI window
WINDOW_RESIZABLE = (False, False)  # Whether the window can be resized
BACKGROUND_IMAGE_NAME = "background.jpg"  # Name of the background image file
BACKGROUND_SIZE = (300, 300)  # Size to which the background image will be resized
BUTTON_TEXT_REAL_TIME = "Real-Time Face Recognition"  # Text for the real-time recognition button
BUTTON_TEXT_VIDEO_FILE = "Video File Face Recognition"  # Text for the video file recognition button
BUTTON_TEXT_ADD_PHOTOS = "Add Photos"  # Text for the add photos button
BUTTON_X_POSITION = 150  # X position for all buttons on the canvas
BUTTON_Y_POSITION_REAL_TIME = 125  # Y position for the real-time recognition button
BUTTON_Y_POSITION_VIDEO_FILE = 175  # Y position for the video file recognition button
BUTTON_Y_POSITION_ADD_PHOTOS = 225  # Y position for the add photos button


def start_gui(known_encodings, known_filenames, people_directory):
    # Initialize the main window
    root = Tk()
    root.title(WINDOW_TITLE)
    root.geometry(WINDOW_SIZE)
    root.resizable(*WINDOW_RESIZABLE)

    # Load and display background image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bg_image_path = os.path.join(script_dir, BACKGROUND_IMAGE_NAME)
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize(BACKGROUND_SIZE, Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = Canvas(root, width=BACKGROUND_SIZE[0], height=BACKGROUND_SIZE[1])
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Add buttons on top of the background
    btn_real_time = Button(root, text=BUTTON_TEXT_REAL_TIME, command=lambda: real_time_recognition(known_encodings, known_filenames))
    btn_video_file = Button(root, text=BUTTON_TEXT_VIDEO_FILE, command=lambda: video_file_recognition(known_encodings, known_filenames))
    btn_add_photos = Button(root, text=BUTTON_TEXT_ADD_PHOTOS, command=lambda: add_photos(people_directory))

    # Place buttons on the canvas
    canvas.create_window(BUTTON_X_POSITION, BUTTON_Y_POSITION_REAL_TIME, window=btn_real_time)
    canvas.create_window(BUTTON_X_POSITION, BUTTON_Y_POSITION_VIDEO_FILE, window=btn_video_file)
    canvas.create_window(BUTTON_X_POSITION, BUTTON_Y_POSITION_ADD_PHOTOS, window=btn_add_photos)

    # Start the GUI event loop
    root.mainloop()
