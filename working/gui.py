import os
from tkinter import Tk, Button, Canvas, filedialog, simpledialog
from PIL import Image, ImageTk
from Face_Recog import real_time_recognition, video_file_recognition, add_photos

# Constants
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 300
BUTTON_X = 150
BUTTON_Y_REAL_TIME = 125
BUTTON_Y_VIDEO_FILE = 175
BUTTON_Y_ADD_PHOTOS = 225
BG_IMAGE_PATH = "background.jpg"

def start_gui(known_encodings, known_filenames, people_directory):
    root = Tk()
    root.title("Face Recognition")
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    root.resizable(False, False)

    # Load and display background image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bg_image_path = os.path.join(script_dir, BG_IMAGE_PATH)
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((WINDOW_WIDTH, WINDOW_HEIGHT), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Add buttons on top of the background
    btn_real_time = Button(root, text="Real-Time Face Recognition", command=lambda: real_time_recognition(known_encodings, known_filenames))
    btn_video_file = Button(root, text="Video File Face Recognition", command=lambda: video_file_recognition(known_encodings, known_filenames))
    btn_add_photos = Button(root, text="Add Photos", command=lambda: add_photos(people_directory))

    # Place buttons on the canvas
    canvas.create_window(BUTTON_X, BUTTON_Y_REAL_TIME, window=btn_real_time)
    canvas.create_window(BUTTON_X, BUTTON_Y_VIDEO_FILE, window=btn_video_file)
    canvas.create_window(BUTTON_X, BUTTON_Y_ADD_PHOTOS, window=btn_add_photos)

    root.mainloop()
