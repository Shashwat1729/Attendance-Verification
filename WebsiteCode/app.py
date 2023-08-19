from flask import Flask, render_template, request
import os
import cv2
from mtcnn import MTCNN
import numpy as np
import uuid
import time
import glob
import os

app = Flask(__name__)
person_count = 0

@app.before_request
def delete_old_images_on_startup():
    delete_old_images()


@app.route('/delete_old_images', methods=['POST'])
def delete_old_images_route():
    delete_old_images()
    return 'Old images deleted successfully.'

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    if request.method == 'POST':
        if 'single_image' in request.files:
            image = request.files['single_image']
            if image and is_image_above_50kb(image):
                images_data = process_single_image(image)
                return render_template('index.html', images=images_data)
            else:
                error_message = "Please upload an image larger than 50KB."

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
                error_message = f"Please upload images larger than 50KB. Invalid images: {invalid_images_str}"
            if len(valid_images)>0:
                images_data = process_bulk_images(valid_images)
                return render_template('index.html', images=images_data,error_message=error_message)
    return render_template('index.html', images=[], error_message=error_message)


def is_image_above_50kb(image):
    image.seek(0, os.SEEK_END)
    size = image.tell()
    image.seek(0)
    return size > 50000

def process_single_image(image):
    i = 0
    images_data = []
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    count, processed_images = count_people(img, i)
    images_data.append((processed_images, count))

    delete_old_images()  # Delete old images

    return images_data


def process_bulk_images(images):
    images_data = []
    for i, image in enumerate(images):
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        count, processed_images = count_people(img, i)
        images_data.append((processed_images, count))

    delete_old_images()

    return images_data



def count_people(image, i):
    images_data = []
    detector = MTCNN()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    for face in results:
        if face['confidence'] > 0.6:
            x, y, width, height = face['box']
            cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
    count = len(results)

    random_name = str(uuid.uuid4())
    img_path = f"static/{random_name}.jpg"
    cv2.imwrite(img_path, image)
    images_data.append(img_path)

    return count, img_path

def delete_old_images():
    images_folder = "static"
    current_time = time.time()
    for file_path in glob.glob(f"{images_folder}/*.jpg"):
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            if current_time - file_creation_time > 15 * 24 * 60 * 60:
                os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
