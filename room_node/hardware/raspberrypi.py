from typing import Optional, Callable

from gpiozero import LED as GPIO_LED, InputDevice, Button


class GpioLED:
    def __init__(self, led_pin: int):
        self._led = GPIO_LED(led_pin)

    def turn_on(self):
        self._led.on()

    def turn_off(self):
        self._led.off()

    def blink(self, period_on: float, period_off: float, times: Optional[int] = None):
        self._led.blink(period_on, period_off, times)


class DoorGPIO:
    def __init__(self, door_lock_pin: int, door_check_opened_pin: int, closing_delay: float,
                 lock_kwargs: dict[str, ...] = None, checker_kwargs: dict[str, ...] = None):
        lock_kwargs = lock_kwargs or {}
        checker_kwargs = checker_kwargs or {}

        self._closing_delay = closing_delay
        # Используется класс LED, а не OutputDevice т.к. удобно использовать метод .blink для открытия замка
        self._door_lock = GPIO_LED(door_lock_pin, **lock_kwargs)
        self._door_checker = InputDevice(door_check_opened_pin, **checker_kwargs)

    @property
    def is_open(self) -> bool:
        return bool(self._door_checker.value)

    def open(self) -> None:
        """Открыть замок на время closing_delay"""
        self._door_lock.blink(on_time=self._closing_delay, off_time=0.0, n=1)

    def close(self) -> None:
        self._door_lock.off()


class ButtonGPIO:
    def __init__(self, button_pin: int, hold_time: float,
                 button_kwargs: Optional[dict[str, ...]] = None):
        button_kwargs = button_kwargs or {}
        self._button = Button(button_pin, hold_time=hold_time, **button_kwargs)

    def set_click_callback(self, callback: Callable) -> None:
        self._button.when_activated = callback

    def set_hold_callback(self, callback: Callable) -> None:
        self._button.when_held = callback

