import time
from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol

from transliterate import translit

from face_recognition import NumpyImage, Descriptor, Recognizer
from face_recognition.two_step import FaceImageNormalizer

from room_node.entities.tasks import TaskType, TaskResult, DeferredTask, Task, TaskStatus
from room_node.entities.access_control import User, Visit, AccessCheck
from room_node import text

from room_node.hardware.devices_protocols import Hardware
from room_node.tasks.tasker import Tasker


class MainNodeConnection(Protocol):
    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def check_access_by_face(self, normalized_image: NumpyImage) -> AccessCheck: ...

    @abstractmethod
    def check_access_by_descriptor(self, descriptor: Descriptor) -> AccessCheck: ...

    @abstractmethod
    def report_visit(self, visit: Visit) -> None: ...

    @abstractmethod
    def report_task_performed(self, result: TaskResult) -> None: ...

    # TODO: Заменить на установку колбэка на получение задач
    @abstractmethod
    def get_tasks(self) -> list[Task]: ...


TASKS_PERIOD = 2.0
ACCESS_CONTROL_PERIOD = 0.01


class RoomNode:
    def __init__(self, hardware: Hardware,
                 main_node_connection: MainNodeConnection,
                 face_image_normalizer: FaceImageNormalizer,
                 recognizer: Recognizer,
                 tasker: Tasker):
        self._hardware = hardware
        self._main_node_conn = main_node_connection
        self._face_image_normalizer = face_image_normalizer
        self._recognizer = recognizer
        self._tasker = tasker

    def start(self):
        self._setup_hardware()
        self._setup_task_handlers()
        self._main_node_conn.initialize()
        self.loop()

    def _setup_hardware(self):
        time.sleep(5)
        self._not_open_door()

    def _setup_task_handlers(self):
        self._tasker.setup_handlers({
            TaskType.SCHEDULE_TASK: self.__task_schedule_task,
            TaskType.CANCEL_TASK: self.__task_cancel_task,
            TaskType.OPEN_DOOR: self.__task_open_door,
        })

    def loop(self):
        next_access_control = next_tasks = 0.0
        while True:
            current = time.perf_counter()

            try:
                if next_access_control < current:
                    self._access_control()
                    next_access_control = current + ACCESS_CONTROL_PERIOD

                if next_tasks < current:
                    self._tasks()
                    next_tasks = current + TASKS_PERIOD
            except Exception as e:
                print(e)
                continue

    def _access_control(self):
        # Face detection and normalization
        image = self._hardware.camera.get_image()
        normalized_image = self._face_image_normalizer.normalize(image)
        if normalized_image is None:
            return
        descriptor = self._recognizer.extract_features(normalized_image)
        access_check = self._main_node_conn.check_access_by_descriptor(descriptor)
        if access_check is None:
            return
        if access_check.is_known and access_check.accessed:
            self._open_door(access_check.user)
        else:
            self._not_open_door()

    def _tasks(self):
        new_tasks = self._main_node_conn.get_tasks()
        results = [
            (task.id, self._tasker.perform(task))
            for task in new_tasks
        ]
        results += self._tasker.perform_suitable_tasks()
        for id_, result in results:
            self._main_node_conn.report_task_performed(result)

    def _open_door(self, user: User = None):
        name_line = translit(f"{user.surname} {user.name[0]}. {user.patronymic[0]}.", 'ru', True) if user else ''
        self._hardware.display.update_text(text.ACCESSED, name_line)
        self._hardware.door.open(4)
        time.sleep(4)
        self._hardware.display.update_text(*text.WAITING_VISITORS)
        if user is not None:
            self._main_node_conn.report_visit(
                visit=Visit(user.id, datetime.now().astimezone())
            )

    def _not_open_door(self):
        self._hardware.display.update_text(*text.YOU_ARE_NOT_ACCESSED)
        time.sleep(0.5)
        self._hardware.display.update_text(*text.WAITING_VISITORS)

    #######################################################################################
    #  Task Handlers
    #######################################################################################

    def __task_schedule_task(self, kwargs: dict[str, Any]):
        """
        Добавить отложенную задачу.
        kwargs: {task: Task as dict, after: datetime, before: datetime}
        """
        task = DeferredTask(**kwargs['task'], after=kwargs['after'], before=kwargs['before'])
        self._tasker.schedule_task(task)
        return TaskStatus.DONE

    def __task_cancel_task(self, kwargs: dict[str, Any]):
        """
        Отменить отложенную задачу.
        kwargs: {task_id: int}
        """
        task_id = kwargs['task_id']
        self._tasker.cancel_task(task_id)
        self._main_node_conn.report_task_performed(TaskResult(task_id, TaskStatus.CANCELLED))
        return TaskStatus.DONE

    def __task_open_door(self, _):
        """
        Открыть дверь прямо сейчас.
        kwargs: {}
        """
        self._open_door()
        return TaskStatus.DONE
