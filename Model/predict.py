import tempfile

from flask import Flask, jsonify, request
import os
import cv2
import keras
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Создается список категорий (CATEGORIES), каждый элемент которого соответствует определенному классу животного.
CATEGORIES = ['hedgehog', 'weasel']

CURR_DIR = os.path.dirname('Save_model/animals-prediction-23.04.25')

MODEL_NAME = 'animals-prediction-23.04.25'

MODEL_PATH = os.path.join(CURR_DIR, MODEL_NAME)


# Определяется функция image, которая считывает изображение из указанного пути, изменяет его размер до 200x200 пикселей
# и возвращает массив numpy.
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (200, 200))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 200, 200)
    return new_arr


# Загружается предварительно обученная модель (model), которая будет использоваться для предсказания класса изображения.
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)


@app.route('/predict', methods=['POST'])
def predict():
    # # Получение файла из объекта запроса
    photo = request.files.get('photo')
    # # Получение подписи из объекта запроса
    signature = request.form.get('signature')

    # Сохранение файла во временный файл
    filename = secure_filename(photo.filename)
    temp_file_path = os.path.join(tempfile.gettempdir(), filename)
    # print({'signature': signature})
    try:
        with open(temp_file_path, 'wb') as f:
            f.write(photo.read())

        # Выполняется предсказание класса изображения с помощью модели model и функции image для указанного изображения.
        if not os.path.exists(temp_file_path):
            return jsonify({'error': f"File {temp_file_path} does not exist"}), 400
        else:
            image_arr = load_image(temp_file_path)
            prediction = model.predict(image_arr)
            predicted_class_index = prediction.argmax()
            predicted_class_name = CATEGORIES[predicted_class_index]
            # Результат предикта
            print({'predicted_class': predicted_class_name})
            return jsonify({"result": predicted_class_name})
            # return jsonify({"result": "Hello "})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# @app.route('/users', methods=['POST'])
# def getUser():
#     return jsonify({"result": "1"})


@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
