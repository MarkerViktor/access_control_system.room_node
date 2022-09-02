from dataclasses import dataclass
from datetime import datetime as Datetime
from enum import Enum

from typing import Any


class TaskType(str, Enum):
    """Тип задачи для узла помещения."""
    OPEN_DOOR = 'OPEN_DOOR'
    SCHEDULE_TASK = 'SCHEDULE_TASK'
    CANCEL_TASK = 'CANCEL_TASK'


class TaskStatus(str, Enum):
    DONE = 'DONE'
    CANCELLED = 'CANCELLED'
    PERFORMING_ERROR = 'PERFORMING_ERROR'
    KWARGS_ERROR = 'PARAMETERS_ERROR'
    NO_SUITABLE_HANDLER_ERROR = 'NO_SUITABLE_HANDLER_ERROR'


@dataclass
class TaskResult:
    """Результат выполнения задачи."""
    task_id: int
    status: TaskStatus


@dataclass
class Task:
    """Задача для узла помещения."""
    id: int
    type: TaskType
    kwargs: dict[str, Any]

    def __repr__(self):
        return f'Task(id={self.id},type={self.type})'


@dataclass
class DeferredTask(Task):
    """Отложенная задача узла помещения."""
    after: Datetime
    before: Datetime

    def is_valid(self, moment: Datetime):
        return self.after <= moment <= self.before

    @classmethod
    def from_task(cls, task: Task, time_period: tuple[Datetime, Datetime]):
        return cls(
            id=task.id,
            type=task.type,
            kwargs=task.kwargs,
            after=time_period[0],
            before=time_period[1],
        )
