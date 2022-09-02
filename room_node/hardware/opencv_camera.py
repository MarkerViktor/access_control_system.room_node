import threading
from typing import Iterator, Union

import cv2
from numpy import uint8
from numpy.typing import NDArray
from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB

def fast_grab(cap: VideoCapture):
    while True:
        cap.grab()

class CV2Camera:
    def __init__(self, *capture_params: Union[str, int], show: bool):
        self._image_stream = self._cv2_capture(capture_params, show)

    def get_image(self) -> NDArray[uint8]:
        image = next(self._image_stream)
        return cvtColor(image, COLOR_BGR2RGB)

    @staticmethod
    def _cv2_capture(capture_params, show: bool) -> Iterator[NDArray[uint8]]:
        cap = VideoCapture(*capture_params)
        threading.Thread(target=fast_grab, args=(cap,)).start()
        try:
            while cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                result, image = cap.retrieve()
                if result:
                    if show:
                        cv2.imshow('cam', image)
                        cv2.pollKey()
                    yield cvtColor(image, COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Can't capture image from cam (params={capture_params!r}).")
        finally:
            cap.release()
            cv2.destroyAllWindows()
