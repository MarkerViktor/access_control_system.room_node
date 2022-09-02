import logging
import time
from http import HTTPStatus
from io import BytesIO
from typing import Optional, Any

import requests
from PIL import Image

from face_recognition import Descriptor, NumpyImage
from room_node.entities.access_control import User, Visit, AccessCheck
from room_node.entities.tasks import Task, TaskResult
from room_node.room_node import MainNodeConnection

logger = logging.getLogger('MAIN_NODE_HTTP_CONNECTION')

def except_requests_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except requests.ConnectionError:
            raise Exception('Подключение не установлено')
    return wrapper

class MainNodeHTTPConnection(MainNodeConnection):
    def __init__(self, main_node_host: str, login_token: str):
        self._host_url = f"http://{main_node_host}/"
        self._login_token = login_token
        self._session = requests.Session()
        self._session.hooks = {'response': self._check_unauthorized_callback}

    def check_access_by_face(self, normalized_image: NumpyImage) -> Optional[AccessCheck]:
        data = self._post('access/check_by_face',
                          files={'image': _numpy_image_as_file(normalized_image)})
        return _access_check_deserialize(data['result']) if data['success'] else None

    def check_access_by_descriptor(self, descriptor: Descriptor) -> Optional[AccessCheck]:
        data = self._post('access/check_by_descriptor',
                          data={'features': list(descriptor)})
        return _access_check_deserialize(data['result']) if data['success'] else None

    def report_visit(self, visit: Visit) -> Optional[int]:
        data = self._post('access/report_visit',
                          data={'user_id': visit.user_id, 'datetime': visit.datetime.astimezone().isoformat()},)
        return data['result']['visit_id'] if data['success'] else None

    def get_tasks(self) -> list[Task]:
        data = self._get('tasks/undone')
        return [Task(**t) for t in data['result']['tasks']] if data['success'] else []

    def report_task_performed(self, result: TaskResult) -> None:
        self._post('task/report',
                   data={'task_id': result.task_id, 'new_status': result.status})

    def authorize(self, login_token: str) -> Optional[str]:
        data = self._post("authorization/room_login",
                          headers={'Login-Token': login_token})
        return data['result']['temp_token'] if data['success'] else None

    def initialize(self) -> None:
        while True:
            if token := self.authorize(self._login_token):
                self._session.headers['Room-Token'] = token
                break
            else:
                logger.error("Can't login")
                time.sleep(5.0)
        logger.info('Initialized')

    def _check_unauthorized_callback(self, r: requests.Response, *args, **kwargs):
        if r.status_code == HTTPStatus.UNAUTHORIZED:
            self.initialize()
            return self._session.send(r.request)
        return r

    @except_requests_exceptions
    def _post(self, url: str, data: dict[str, Any] = None,
              files: dict[str, BytesIO] = None, headers: dict[str, str] = None) -> dict[str, Any]:
        url = self._host_url + url
        response = self._session.post(url, json=data, files=files, headers=headers)
        return response.json()

    @except_requests_exceptions
    def _get(self, url: str, params: dict[str, str] = None) -> dict[str, Any]:
        url = self._host_url + url
        response = self._session.get(url, params=params)
        return response.json()


def _numpy_image_as_file(image: NumpyImage) -> BytesIO:
    virtual_file = BytesIO()
    Image.fromarray(image).save(virtual_file, 'JPEG')
    virtual_file.seek(0)
    return virtual_file


def _access_check_deserialize(data: dict) -> AccessCheck:
    is_known, accessed = data['is_known'], data['have_access']
    user = User(**data['user']) if is_known else None
    return AccessCheck(is_known, accessed, user)

