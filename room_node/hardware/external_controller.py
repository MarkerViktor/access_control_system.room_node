import serial


class ExternalController:
    def __init__(self, port: serial.Serial):
        self._serial = port

    def update_display(self, symbols: bytes) -> None:
        assert len(symbols) == 32
        command = b'$U' + symbols
        self._serial.write(command)

    def open_door(self, open_period_seconds: int) -> None:
        assert 0 < open_period_seconds < 256
        command = b'$O' + open_period_seconds.to_bytes(1, byteorder="big") + b' ' * 31
        self._serial.write(command)


class DoorExternalController:
    def __init__(self, controller: ExternalController):
        self._controller = controller

    def open(self, period_seconds: int) -> None:
        self._controller.open_door(period_seconds)


class TextDisplayExternalController:
    def __init__(self, controller: ExternalController):
        self._controller = controller

    def update_text(self, first_line: str = "", second_line: str = "") -> None:
        text_raw = f'{first_line[:16]:<16}{second_line[:16]:<16}'.encode('ascii')
        self._controller.update_display(text_raw)
