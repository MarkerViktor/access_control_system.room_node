import logging
from abc import abstractmethod
from datetime import datetime as Datetime
from typing import Protocol, Any

from room_node.entities.tasks import Task, DeferredTask, TaskType, TaskResult, TaskStatus


logger = logging.getLogger("TASKER")
logger.setLevel(logging.DEBUG)


class TaskStorage(Protocol):
    @abstractmethod
    def add(self, task: DeferredTask) -> None:
        """Добавить задачу в хранилище."""
        ...

    @abstractmethod
    def delete(self, task_id: int) -> None:
        """Удалить задачу из хранилища."""
        ...

    @abstractmethod
    def get_all(self) -> list[DeferredTask]:
        """Получить все сохранённые задачи."""

    @abstractmethod
    def take_suitable_tasks(self, after_timestamp: Datetime) -> list[DeferredTask]:
        """Получить задачи, которые нужно выполнить после заданного момента и удалить их."""
        ...


class TaskHandler(Protocol):
    """
    Обработчик определённого типа задач.
    Все исключения должны отлавливаться внутри обработчика,
    должен возвращаться результат со соответствующим TaskStatus."""
    def __call__(self, kwargs: dict[str, Any]) -> TaskStatus: ...


class Tasker:
    def __init__(self, storage: TaskStorage):
        self._storage = storage
        self._handlers: dict[str, TaskHandler] = {}

    def setup_handlers(self, types_handlers: dict[TaskType, TaskHandler]) -> None:
        for type_, handler in types_handlers.items():
            self._handlers[type_] = handler

    def perform(self, task: Task) -> TaskResult:
        """
        Выполнить задачу с помощью заданного ранее обработчика.
        Все исключения должны отлавливаться внутри обработчиков, должен возвращаться
        результат со соответствующим TaskStatus.
        """
        if handler := self._handlers.get(task.type):
            status = handler(task.kwargs)
            if status == TaskStatus.DONE:
                logger.info(f"DONE successfully {task}")
            return TaskResult(task.id, status)
        else:
            logger.error(f'No handler for {task}')
            return TaskResult(task.id, TaskStatus.NO_SUITABLE_HANDLER_ERROR)

    def schedule_task(self, task: DeferredTask) -> None:
        """Запланировать выполнение отложенной задачи"""
        self._storage.add(task)
        logger.info(f"{task} was scheduled")

    def cancel_task(self, task_id: int) -> None:
        """Отменить задачу по идентификатору."""
        self._storage.delete(task_id)
        logger.info(f"Task(id={task_id}) was cancelled")

    def perform_suitable_tasks(self) -> list[TaskResult]:
        """
        Выполнить задачи, для которых время уже пришло.
        Возвращает список результатов выполнения задач.
        """
        now = Datetime.now()
        tasks = self._storage.take_suitable_tasks(now)
        results = [self.perform(task) for task in tasks if task.is_valid(now)]
        return results
