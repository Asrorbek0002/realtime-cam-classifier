import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC


class Model:

    def __init__(self):
        self.model = LinearSVC(max_iter=5000, dual='auto')
        self.is_trained = False

    def train_model(self, counters):
        img_list = []
        class_list = []

        # Load Class 1 images
        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv.resize(img, (150, 150)).flatten()
            img_list.append(img)
            class_list.append(1)

        # Load Class 2 images
        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv.resize(img, (150, 150)).flatten()
            img_list.append(img)
            class_list.append(2)

        if len(img_list) == 0:
            print("Not enough images for training!")
            return

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        print("Training shape:", img_list.shape)
        self.model.fit(img_list, class_list)
        self.is_trained = True
        print("Model trained successfully!")

    def predict(self, frame):
        if not self.is_trained:
            return None

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, (150, 150))
        img = gray.flatten()

        prediction = self.model.predict([img])
        return prediction[0]