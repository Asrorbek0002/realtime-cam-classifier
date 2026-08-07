import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import camera
import model

class App:
    def __init__(self, window=None, window_title="Camera Classifier"):
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
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("1-Sinf", "Birinchi sinf nomini kiriting:", parent=self.window) or "Class 1"
        self.classname_two = simpledialog.askstring("2-Sinf", "Ikkinchi sinf nomini kiriting:", parent=self.window) or "Class 2"

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=self.train)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        if not self.model.is_trained:
            messagebox.showwarning("Ogohlantirish", "Avval modelni o'qiting (Train Model)!")
            return
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            return

        if not os.path.exists('1'):
            os.mkdir('1')        
        if not os.path.exists('2'):
            os.mkdir('2')

        file_path = f'{class_num}/frame{self.counters[class_num - 1]}.jpg'
        cv.imwrite(file_path, cv.cvtColor(frame, cv.COLOR_RGB2GRAY))   
        
        img = PIL.Image.open(file_path)
        img.thumbnail((150, 150), PIL.Image.Resampling.LANCZOS)
        img.save(file_path) 

        self.counters[class_num - 1] += 1

    def train(self):
        self.model.train_model(self.counters)
        if self.model.is_trained:
            messagebox.showinfo("Muvaffaqiyatli", "Model o'qitildi!")

    def reset(self):
        for directory in ['1', '2']:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file) 
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

        self.counters = [1, 1]
        self.model = model.Model()           
        self.auto_predict = False
        self.class_label.config(text='CLASS')
        messagebox.showinfo("Reset", "Barcha ma'lumotlar tozalandi!")

    def update(self):
        if self.auto_predict:
            self.predict()     

        ret, frame = self.camera.get_frame()

        if ret and frame is not None:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)    

    def predict(self):
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            return

        prediction = self.model.predict(frame)

        if prediction == 1:
            self.class_label.config(text=self.classname_one)
            return self.classname_one
        elif prediction == 2:
            self.class_label.config(text=self.classname_two)
            return self.classname_two