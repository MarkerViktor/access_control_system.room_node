from dataclasses import dataclass
from typing import Optional
from datetime import datetime as Datetime


@dataclass
class User:
    """Пользователь системы контроля доступа."""
    id: int
    name: str
    surname: str
    patronymic: str
    position: Optional[str] = None


@dataclass
class Visit:
    """Посещение помещения."""
    user_id: int
    datetime: Datetime


@dataclass
class AccessCheck:
    """Результат проверки доступа."""
    is_known: bool
    accessed: bool
    user: Optional[User] = None
