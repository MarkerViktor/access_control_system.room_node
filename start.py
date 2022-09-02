import logging

import serial
import cv2

from face_recognition.backends.dlib_ import DlibDetector, DlibNormalizer
from face_recognition.two_step import FaceImageNormalizer
from room_node.hardware.external_controller import TextDisplayExternalController, ExternalController, \
    DoorExternalController
from room_node.room_node import RoomNode
from room_node.hardware.devices_protocols import Hardware
from room_node.hardware.opencv_camera import CV2Camera
from room_node.main_node_http_connection import MainNodeHTTPConnection
from room_node.tasks.tasker import Tasker
from room_node.tasks.sqlite_task_storage import SQLiteTaskStorage


logging.basicConfig(level=logging.INFO)


def main():
    controller = ExternalController(
        port=serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=9600,
        )
    )
    RoomNode(
        hardware=Hardware(
            display=TextDisplayExternalController(
                controller=controller,
            ),
            door=DoorExternalController(
                controller=controller,
            ),
            camera=CV2Camera("/dev/video0", show=False),
        ),
        main_node_connection=MainNodeHTTPConnection(
            main_node_host='192.168.1.11:8080',
            login_token='c6646b524a5624fc4329e1af7613ace6',
        ),
        face_image_normalizer=FaceImageNormalizer(
            detector=DlibDetector(),
            normalizer=DlibNormalizer(),
        ),
        tasker=Tasker(
            storage=SQLiteTaskStorage('db.sqlite')
        )
    ).start()


if __name__ == '__main__':
    main()
