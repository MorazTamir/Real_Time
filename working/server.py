# # working/server.py
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from concurrent.futures import ThreadPoolExecutor
# import cv2 as cv
# from io import BytesIO
# from tkinter import Tk, Button
# from tkinter.filedialog import askopenfilename
# from algorithms import load_known_encodings, process_frame

# app = FastAPI()

# # Load known encodings
# people_directory = "../CelebrityTest"  # Update this path to your directory
# known_encodings, known_filenames = load_known_encodings(people_directory)

# executor = ThreadPoolExecutor(max_workers=2)

# @app.post("/process_video/")
# async def process_video(file: UploadFile = File(...), x_frame_interval: int = 1):
#     video_data = await file.read()
#     video_stream = BytesIO(video_data)
#     video_capture = cv.VideoCapture(video_stream)

#     frame_count = 0
#     results = []

#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % x_frame_interval == 0:
#             future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
#             frame_results = future.result()
#             results.append(frame_results)

#     video_capture.release()
#     return JSONResponse(content={"results": results})

# # Real-time face recognition
# @app.get("/real_time_recognition/")
# def real_time_recognition():
#     video_capture = cv.VideoCapture(0)
#     frame_count = 0
#     executor = ThreadPoolExecutor(max_workers=2)

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % 1 == 0:  # Adjust the interval to process every frame
#             future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
#             processed_frame = future.result()
#             cv.imshow('Face Recognition', processed_frame)
#         else:
#             cv.imshow('Face Recognition', frame)

#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv.destroyAllWindows()

# # Video file face recognition
# @app.get("/video_file_recognition/")
# def video_file_recognition():
#     Tk().withdraw()
#     video_path = askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
#     if not video_path:
#         return

#     video_capture = cv.VideoCapture(video_path)
#     frame_count = 0
#     executor = ThreadPoolExecutor(max_workers=2)

#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % 1 == 0:  # Adjust the interval to process every frame
#             future = executor.submit(process_frame, frame.copy(), known_encodings, known_filenames)
#             processed_frame = future.result()
#             cv.imshow('Face Recognition', processed_frame)
#         else:
#             cv.imshow('Face Recognition', frame)

#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

#         for _ in range(30):  # Skip 30 frames ahead
#             video_capture.grab()

#     video_capture.release()
#     cv.destroyAllWindows()

# # Main function
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
