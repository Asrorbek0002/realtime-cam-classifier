from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv


class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):

        img_list = []
        class_list = []

        # Class 1
        for i in range(1, counters[0]):

            img = cv.imread(
                f'1/frame{i}.jpg',
                cv.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = cv.resize(img, (150, 150))
            img = img.flatten()

            img_list.append(img)
            class_list.append(1)

        # Class 2
        for i in range(1, counters[1]):

            img = cv.imread(
                f'2/frame{i}.jpg',
                cv.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = cv.resize(img, (150, 150))
            img = img.flatten()

            img_list.append(img)
            class_list.append(2)

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        print("Training shape:", img_list.shape)

        self.model.fit(img_list, class_list)

        print("Model successfully trained!")

    def predict(self, frame):

        frame = frame[1]

        gray = cv.cvtColor(
            frame,
            cv.COLOR_RGB2GRAY
        )

        gray = cv.resize(
            gray,
            (150, 150)
        )

        img = gray.flatten()

        prediction = self.model.predict([img])

        return prediction[0]