import os
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2 as cv

import camera
import model
import PIL.Image
import PIL.ImageTk


class App:

    def __init__(self, window=None, window_title="CamVision AI"):
        if window is None:
            self.window = tk.Tk()
        else:
            self.window = window

        self.window.title(window_title)

        self.counters = [1, 1]
        self.model = model.Model()
        self.auto_predict = False
        self.camera = camera.Camera()

        self.init_gui()

        self.delay = 15
        self.update()

        self.window.attributes('-topmost', True)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def on_close(self):
        # Release the webcam instead of leaving it locked after the window goes
        self.camera.release()
        self.window.destroy()

    def init_gui(self):
        self.canvas = tk.Canvas(
            self.window, width=self.camera.width, height=self.camera.height
        )
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(
            self.window,
            text="Auto Prediction",
            width=50,
            command=self.auto_predict_toggle,
        )
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = (
            simpledialog.askstring(
                "Class One",
                "Enter the name of the first class:",
                parent=self.window,
            )
            or "Class 1"
        )
        self.classname_two = (
            simpledialog.askstring(
                "Class Two",
                "Enter the name of the second class:",
                parent=self.window,
            )
            or "Class 2"
        )

        self.btn_class_one = tk.Button(
            self.window,
            text=self.classname_one,
            width=50,
            command=lambda: self.save_for_class(1),
        )
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(
            self.window,
            text=self.classname_two,
            width=50,
            command=lambda: self.save_for_class(2),
        )
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(
            self.window, text="Train Model", width=50, command=self.train
        )
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(
            self.window, text="Predict", width=50, command=self.predict
        )
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(
            self.window, text="Reset", width=50, command=self.reset
        )
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

        self.update_sample_count()

    def auto_predict_toggle(self):
        if not self.model.is_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            return

        directory = model.class_dir(class_num)
        os.makedirs(directory, exist_ok=True)

        # Store exactly what the classifier will be fed - resizing here and
        # again at predict time is what silently hurt accuracy before
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, model.IMG_SIZE)

        file_path = os.path.join(directory, f'frame{self.counters[class_num - 1]}.jpg')
        cv.imwrite(file_path, gray)

        self.counters[class_num - 1] += 1
        self.update_sample_count()

    def update_sample_count(self):
        # Show how many samples each class has, so the user knows when to train
        one, two = self.counters[0] - 1, self.counters[1] - 1
        self.btn_class_one.config(text=f"{self.classname_one}  ({one})")
        self.btn_class_two.config(text=f"{self.classname_two}  ({two})")

    def train(self):
        success, message = self.model.train_model(self.counters)
        if success:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showwarning("Cannot train", message)

    def reset(self):
        for class_num in (1, 2):
            directory = model.class_dir(class_num)
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

        self.counters = [1, 1]
        self.model = model.Model()
        self.auto_predict = False
        self.class_label.config(text='CLASS')
        self.update_sample_count()
        messagebox.showinfo("Reset", "All data cleared successfully!")

    def update(self):
        # One grab per cycle - the frame shown is the frame classified
        ret, frame = self.camera.get_frame()

        if ret and frame is not None:
            if self.auto_predict:
                self.predict(frame)

            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame)
            )
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self, frame=None):
        manual = frame is None

        if not self.model.is_trained:
            if manual:
                messagebox.showwarning("Warning", "Please train the model first!")
            return None

        if frame is None:
            ret, frame = self.camera.get_frame()
            if not ret or frame is None:
                return None

        prediction = self.model.predict(frame)

        if prediction == 1:
            self.class_label.config(text=self.classname_one)
            return self.classname_one
        elif prediction == 2:
            self.class_label.config(text=self.classname_two)
            return self.classname_two
        return None