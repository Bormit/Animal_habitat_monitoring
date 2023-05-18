import os
import cv2
import keras
import numpy as np

# Создается список категорий (CATEGORIES), каждый элемент которого соответствует определенному классу животного.

CATEGORIES = ['hen', 'horse', 'squirrel']

ANIMAL_NAME = 'squirrel'  # Тестирования заданого класса

CURR_DIR = os.path.dirname('Save_model/animals-prediction-23.05.12')

MODEL_NAME = 'animals-prediction-23.05.12'

MODEL_PATH = os.path.join(CURR_DIR, MODEL_NAME)

# Указывается путь к папке, в которой находятся изображения для классификации (PATH).
PATH = f"C:\\Users\\user\\IDE_Projects\\Pycharm_Projects\\Animal_habitat_monitoring\\Model\\Train\\{ANIMAL_NAME}"

f = f"Results_of_predicts\\{ANIMAL_NAME}_false.txt"

# Создается файл для записи результатов предсказания для неправильно классифицированных изображений (horse_false.txt)
# в директории Results_of_predicts.
resultWrite = open(f, 'w')

# Определяется функция image, которая считывает изображение из указанного пути, изменяет его размер до 200x200 пикселей
# и возвращает массив numpy.

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (200, 200))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 200, 200)
    return new_arr


# Загружается предварительно обученная модель (model), которая будет использоваться для предсказания класса изображения.

model = keras.models.load_model(MODEL_PATH)

# Создаются два списка: true - для правильно классифицированных изображений и false - для неправильно
# классифицированных.
true = []
false = []

# Происходит итерация по всем изображениям в указанной директории с помощью функции os.listdir.
for i in os.listdir(PATH):
    # Выполняется предсказание класса изображения с помощью модели model и функции image для каждого изображения в
    # директории.
    prediction = model.predict([image(os.path.join(PATH, i))])
    # Если предсказанный класс соответствует "horse", то имя файла добавляется в список true
    if CATEGORIES[prediction.argmax()] == ANIMAL_NAME:
        true.append([ANIMAL_NAME, os.path.join(PATH, i)])
    # иначе добавляется в список false.
    else:
        false.append([CATEGORIES[prediction.argmax()], os.path.join(PATH, i)])

# Итерируется по списку false и каждый элемент списка записывается в файл.
for i in false:
    resultWrite.write(str(i) + "\n")

resultWrite.close()

try:
    with open(f, 'r') as file:
        lines = file.readlines()
        print(f"Количество строк в файле {f}: {len(lines)}")
except FileNotFoundError:
    print(f"Файл {f} не найден")
except IOError:
    print(f"Не удалось прочитать файл {f}")

