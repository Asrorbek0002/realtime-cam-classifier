import cv2 as cv


class Camera:

    def __init__(self, index=0):
        self.camera = cv.VideoCapture(index)
        if not self.camera.isOpened():
            raise ValueError(f"Unable to open camera {index}!")

        self.width = int(self.camera.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        """Hand the device back to the OS. Safe to call more than once."""
        camera = getattr(self, 'camera', None)
        if camera is not None and camera.isOpened():
            camera.release()

    def __del__(self):
        self.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)
