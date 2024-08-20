import os
from tkinter import Tk, Button, Canvas, filedialog, simpledialog
from PIL import Image, ImageTk
from Face_Recog import real_time_recognition, video_file_recognition, add_photos

def start_gui(known_encodings, known_filenames, people_directory):
    root = Tk()
    root.title("Face Recognition")
    root.geometry("300x300")
    root.resizable(False, False)

    # Load and display background image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bg_image_path = os.path.join(script_dir, "background.jpg")
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((300, 300), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = Canvas(root, width=300, height=300)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Add buttons on top of the background
    btn_real_time = Button(root, text="Real-Time Face Recognition", command=lambda: real_time_recognition(known_encodings, known_filenames))
    btn_video_file = Button(root, text="Video File Face Recognition", command=lambda: video_file_recognition(known_encodings, known_filenames))
    btn_add_photos = Button(root, text="Add Photos", command=lambda: add_photos(people_directory))

    # Place buttons on the canvas
    canvas.create_window(150, 125, window=btn_real_time)
    canvas.create_window(150, 175, window=btn_video_file)
    canvas.create_window(150, 225, window=btn_add_photos)

    root.mainloop()
