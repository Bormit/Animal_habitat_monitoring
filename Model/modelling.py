import datetime
import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

neurons = 2

func_activate = 'softmax'

# путь к директории для сохранения файлов
SAVE_DIR = 'data'

# путь к директории для сохранения модели
SAVE_MODEL = 'Save_model'

# Генерируем уникальное имя для модели
date_today = datetime.datetime.now().date().strftime('%y.%m.%d')
MODEL_DIR_NAME = os.path.join(SAVE_MODEL, f"animals-prediction-{date_today}")

# Создаем объект TensorBoard Callback для записи метрик и логов обучения во время обучения модели в Keras
# и указываем дирректорию 'logs/' для записи логов.
tensorboard = TensorBoard(log_dir='logs/')

# Загружаем данные из файлов X.pkl и y.pkl
try:
    with open(os.path.join(SAVE_DIR, "X.pkl"), 'rb') as f:
        X = pickle.load(f)
    with open(os.path.join(SAVE_DIR, "y.pkl"), 'rb') as f:
        y = pickle.load(f)

        # Нормализуем данные и изменяем размерность массива X
        X = X / 255
        X = X.reshape(-1, 200, 200, 1)

        # Создаем последовательную модель
        model = Sequential()

        # Добавляем сверточные и пулинговые слои
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # model.add(Conv2D(64, (3, 3), activation='relu'))  # закомментированный слой для проведения сравнительного анализа обучаемой модели
        # model.add(MaxPooling2D((2, 2)))

        # Преобразуем данные перед подачей на полносвязные слои
        model.add(Flatten())

        # Добавляем полносвязные слои
        model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))
        # model.add(Dense(128, activation='relu'))  # закомментированный слой для проведения сравнительного анализа обучаемой модели

        # Добавляем выходной слой с функцией активации
        model.add(Dense(neurons, activation=func_activate))

        # Компилируем модель
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Обучаем модель
        model.fit(X, y, epochs=5, validation_split=0.1, batch_size=32, callbacks=[tensorboard])

        # Сохраняем обученную модель в файл
        model.save(MODEL_DIR_NAME)
        print(f"The model was saved under the name", f"animals-prediction-{date_today}")
except IOError:
    print("Не удалось загрузить файлы X.pkl and y.pkl.")

