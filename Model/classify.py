import numpy as np
import cv2
import os
import pickle
import random
from pathlib import Path
from tqdm import tqdm
import argparse

# путь к директории с обучающими изображениями
DIRECTORY = r'Train'

# размер изображений после обработки
IMG_SIZE = 200

# инициализация переменной-счетчика категорий
index = 1

# путь к директории для сохранения файлов
SAVE_DIR = 'data'

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Train an animal recognition neural network')
parser.add_argument('--directory', type=str, default=DIRECTORY, help='path to directory with training images')
parser.add_argument('--img-size', type=int, default=IMG_SIZE, help='size of the processed images')
parser.add_argument('--output-dir', type=str, default='.', help='output directory')
args = parser.parse_args()

# список категорий изображений
CATEGORIES = ['duck', 'grey_owl', 'hedgehog', 'spotted_woodpecker', 'weasel']

data = []  # создание пустого списка для хранения признаков и меток

# цикл по категориям изображений
for index, category in enumerate(CATEGORIES, 1):
    # путь к директории с изображениями текущей категории
    path = Path(args.directory) / category
    # цикл по изображениям текущей категории с отображением прогресса выполнения
    for img_path in tqdm(list(path.glob('*')), desc=f'{index}) Classification {category}s...'):
        label = CATEGORIES.index(category)  # метка класса изображения
        arr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # загрузка изображения
        new_arr = cv2.resize(arr, (args.img_size, args.img_size))  # изменение размера изображения
        data.append([new_arr, label])  # добавление признаков и метки в список признаков и меток

# перемешивание списка признаков и меток случайным образом
random.shuffle(data)

X = np.array([features for features, label in data])  # преобразование списка признаков в массив NumPy
y = np.array([label for features, label in data])

# создаем директорию для сохранения файлов, если она не существует
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# сохраняем массивы X и y в два файла X.pkl и y.pkl соответственно
try:
    with open(os.path.join(SAVE_DIR, "X.pkl"), "wb") as f:
        pickle.dump(X, f)
    with open(os.path.join(SAVE_DIR, "y.pkl"), "wb") as f:
        pickle.dump(y, f)
    print("Classification completed!")
except IOError:
    print("Error saving files X.pkl and y.pkl.")




