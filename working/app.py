# from flask import Flask, request, jsonify
# import os
# import face_recognition as fr

# app = Flask(__name__)

# # Directory to save uploaded photos
# UPLOAD_FOLDER = 'path_to_your_upload_folder'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Route to handle photo uploads
# @app.route('/upload', methods=['POST'])
# def upload_photo():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     # You can add additional processing here if needed

#     return jsonify({'message': 'File uploaded successfully'}), 200

# # Route to handle adding new photos
# @app.route('/add_photos', methods=['POST'])
# def add_photos():
#     files = request.files.getlist('files[]')
#     for file in files:
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # You can add additional processing here if needed

#     return jsonify({'message': 'Photos added successfully'}), 200

# if __name__ == '__main__':
#     app.run(debug=True)
