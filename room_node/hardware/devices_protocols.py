from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, Callable, Optional

from numpy import uint8
from numpy.typing import NDArray


class Door(Protocol):
    """Дверь с электромеханическим замком."""

    @abstractmethod
    def open(self, period_seconds: int): ...


class Camera(Protocol):
    """Камера. Возвращает видео в виде отдельных изображений при вызове .get_image()"""

    @abstractmethod
    def get_image(self) -> NDArray[uint8]: ...


class TextDisplay(Protocol):
    """Текстовый информационный дисплей."""

    @abstractmethod
    def update_text(self, first_line: str = "", second_line: str = "") -> None: ...


class Button(Protocol):
    """Кнопка. Вызывает установленные колбеки."""

    @abstractmethod
    def set_click_callback(self, callback: Callable): ...

    @abstractmethod
    def set_hold_callback(self, callback: Callable): ...


class LED(Protocol):
    """Информационная лампа"""

    @abstractmethod
    def turn_on(self): ...

    @abstractmethod
    def turn_off(self): ...

    @abstractmethod
    def blink(self, period_on: float, period_off: float, times: Optional[int] = None): ...


@dataclass
class LEDPanel:
    ok: LED
    error: LED

    @contextmanager
    def wait_mode(self):
        self.ok.blink(0.2, 0.2)
        try:
            yield
        finally:
            self.ok.turn_off()


@dataclass
class Hardware:
    display: TextDisplay
    door: Door
    camera: Camera
