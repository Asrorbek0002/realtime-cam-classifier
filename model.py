import os

import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC

# Paths are resolved relative to this file so the app works from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Every image the classifier ever sees passes through this size
IMG_SIZE = (150, 150)


def class_dir(class_num):
    """Folder holding the captured samples for one class."""
    return os.path.join(DATA_DIR, str(class_num))


def preprocess(gray):
    """Grayscale image -> the flat feature vector the classifier expects.

    Used for both training samples and live frames, so the two can never drift
    apart.
    """
    return cv.resize(gray, IMG_SIZE).flatten()


class Model:

    def __init__(self):
        self.model = LinearSVC(max_iter=5000, dual='auto')
        self.is_trained = False

    def train_model(self, counters):
        """Fit the classifier on the captured samples.

        Returns (success, message) so the caller can report the outcome.
        """
        img_list = []
        class_list = []

        for class_num in (1, 2):
            for i in range(1, counters[class_num - 1]):
                path = os.path.join(class_dir(class_num), f'frame{i}.jpg')
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_list.append(preprocess(img))
                class_list.append(class_num)

        # LinearSVC raises ValueError on a single class - catch it with a
        # message the user can actually act on
        if len(set(class_list)) < 2:
            self.is_trained = False
            return False, "Capture samples for BOTH classes before training."

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.model.fit(img_list, class_list)
        self.is_trained = True

        counts = [int((class_list == c).sum()) for c in (1, 2)]
        return True, (
            f"Trained on {len(class_list)} samples "
            f"({counts[0]} + {counts[1]}), {img_list.shape[1]} features each."
        )

    def predict(self, frame):
        """Classify one RGB frame. Returns 1, 2, or None if untrained."""
        if not self.is_trained:
            return None

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        return int(self.model.predict([preprocess(gray)])[0])
