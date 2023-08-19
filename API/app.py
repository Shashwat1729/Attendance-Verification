from flask import Flask, request, send_from_directory
import os
import cv2
from mtcnn import MTCNN
import numpy as np
import uuid
import time
import glob
import tempfile


app = Flask(__name__)
person_count = 0

@app.before_request
def delete_old_images_on_startup():
    delete_old_images()

@app.route('/delete_old_images', methods=['POST'])
def delete_old_images_route():
    delete_old_images()
    return 'Old images deleted successfully.'

@app.route('/api/process_images', methods=['POST'])
def process_images():
    images_data = []
    if 'single_image' in request.files:
        image = request.files['single_image']
        if image and is_image_above_50kb(image):
            img_data = image.read()
            count, processed_images = count_people(img_data)
            images_data.append((processed_images, count))
        else:
            return 'Please upload an image larger than 50KB.', 400

    elif 'bulk_images[]' in request.files:
        images = request.files.getlist('bulk_images[]')
        valid_images = []
        invalid_images = []
        for image in images:
            if is_image_above_50kb(image):
                valid_images.append(image)
            else:
                invalid_images.append(image.filename)
        if len(invalid_images) > 0:
            invalid_images_str = ", ".join(invalid_images)
            print (f"Please upload images larger than 50KB. Invalid images: {invalid_images_str}")
        if len(valid_images) > 0:
            for i, image in enumerate(valid_images):
                img_data = image.read()
                count, processed_images = count_people(img_data)
                images_data.append((processed_images, count))

    return {'images_data': images_data},200

def is_image_above_50kb(image):
    image.seek(0, os.SEEK_END)
    size = image.tell()
    image.seek(0)
    return size > 50000

def count_people(image_data):
    images_data = []
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detector = MTCNN()
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    for face in results:
        if face['confidence'] > 0.6:
            x, y, width, height = face['box']
            cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
    count = len(results)

    random_name = str(uuid.uuid4())
    img_path = f"static/{random_name}.jpg"
    cv2.imwrite(img_path, img)
    images_data.append(img_path)

    return count, img_path

@app.route('/static/<path:filename>', methods=['GET'])
def get_annotated_image(filename):
    return send_from_directory('static', filename)

def delete_old_images():
    images_folder = "static"
    current_time = time.time()
    for file_path in glob.glob(f"{images_folder}/*.jpg"):
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            if current_time - file_creation_time > 15 * 24 * 60 * 60:
                os.remove(file_path)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host = '0.0.0.0', port=2000, debug=True)

