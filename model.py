from sklearn.svm import LinearSVC
from sklearn.exceptions import NotFittedError
import numpy as np
import cv2 as cv

class Model:
    def __init__(self):
        self.model = LinearSVC(max_iter=2000)
        self.is_trained = False

    def train_model(self, counters):
        img_list = []
        class_list = []

        # 1-Sinf ma'lumotlarini yuklash
        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv.resize(img, (150, 150)).flatten()
            img_list.append(img)
            class_list.append(1)

        # 2-Sinf ma'lumotlarini yuklash
        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv.resize(img, (150, 150)).flatten()
            img_list.append(img)
            class_list.append(2)

        if len(img_list) == 0:
            print("O'qitish uchun rasmlar yetarli emas!")
            return

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        print("O'qitish shakli:", img_list.shape)
        self.model.fit(img_list, class_list)
        self.is_trained = True
        print("Model muvaffaqiyatli o'qitildi!")

    def predict(self, frame):
        if not self.is_trained:
            return None

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, (150, 150))
        img = gray.flatten()

        prediction = self.model.predict([img])
        return prediction[0]