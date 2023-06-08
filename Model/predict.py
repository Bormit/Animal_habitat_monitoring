import tempfile

from flask import Flask, jsonify, request
import os
import cv2
import keras
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Создается список категорий (CATEGORIES), каждый элемент которого соответствует определенному классу животного.
CATEGORIES = ['duck', 'grey_owl', 'hedgehog', 'spotted_woodpecker', 'weasel']

CURR_DIR = os.path.dirname('Save_model/animals-prediction-23.06.06_test')

MODEL_NAME = 'animals-prediction-23.06.06_test'

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
            # Задайте пороговое значение для вероятности предсказания класса
            threshold = 0.5

            # Проверяем, что вероятность для всех классов меньше порогового значения
            if all(p < threshold for p in prediction[0]):
                return jsonify({"result": "No category matches"})

            else:
                predicted_class_name = CATEGORIES[predicted_class_index]
                # Результат предикта
                if signature == animal_translate(predicted_class_name):
                    print({'predicted_class': animal_translate(predicted_class_name)})
                    return jsonify({"result": animal_translate(predicted_class_name)})
                else:
                    print({'predicted_class': animal_translate(predicted_class_name)})
                    return jsonify({"result": f"На фотографии не {signature}!"})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
def animal_translate(animal):
    if animal == "duck":
        return "Утка"
    if animal == "grey_owl":
        return "Рогатая сова"
    if animal == "hedgehog":
        return "Ёж"
    if animal == "spotted_woodpecker":
        return "Пёстрый дятел"
    if animal == "weasel":
        return "Ласка"



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
    app.run(host='0.0.0.0', port=8000)
