import time
from typing import Optional


class VirtualLED:
    def __init__(self, led_name: str):
        self._name = led_name

    def turn_on(self):
        print(f"Virtual LED «{self._name}» is ON")

    def turn_off(self):
        print(f"Virtual LED «{self._name}» is OFF")

    def blink(self, period_on: float, period_off: float, times: Optional[int] = None):
        print(f"Virtual LED «{self._name}» is blinking (periods=[{period_on}, {period_off}]; times={times})")


class VirtualDoor:
    def __init__(self, name: str, opening_delay_sec: float):
        self._name = name
        self._delay = opening_delay_sec
        self.is_opened = False

    def open(self):
        print(f"Virtual door «{self._name}» is opening.")
        self.is_opened = True
        time.sleep(self._delay)

    def close(self):
        print(f"Virtual door «{self._name}» is closing.")
        self.is_opened = False
