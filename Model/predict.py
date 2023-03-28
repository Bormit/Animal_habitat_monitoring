import os
import cv2
import keras
import numpy as np

# Создается список категорий (CATEGORIES), каждый элемент которого соответствует определенному классу животного.
CATEGORIES = ['hedgehog', 'weasel']

MODEL_NAME = 'animals-prediction-23.03.28'

# Определяется функция image, которая считывает изображение из указанного пути, изменяет его размер до 200x200 пикселей
# и возвращает массив numpy.
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (200, 200))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 200, 200)
    return new_arr


# Загружается предварительно обученная модель (model), которая будет использоваться для предсказания класса изображения.
# название модели у вас можеть быть другой!
model = keras.models.load_model(MODEL_NAME)

# Указывается путь к изображению для которое надо распознать.
image_path = os.path.join("C:", "C:\\Users\\user\\IDE_Projects\\Pycharm_Projects\\Animal_habitat_monitoring\\Parse\\weasel\\000043.jpg")

# Выполняется предсказание класса изображения с помощью модели model и функции image для каждого изображения в
# директории.
if not os.path.exists(image_path):
    print(f"File {image_path} does not exist")
else:
    image_arr = load_image(image_path)
    prediction = model.predict(image_arr)
    predicted_class_index = prediction.argmax()
    predicted_class_name = CATEGORIES[predicted_class_index]
    # Результат предикта
    print(f"Predicted class: {predicted_class_name}")

