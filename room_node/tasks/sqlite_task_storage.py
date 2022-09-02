import sqlite3
import json
import logging
from datetime import datetime as Datetime

from room_node.entities.tasks import DeferredTask
from room_node.tasks.tasker import TaskStorage

logger = logging.getLogger('SQLITE_TASK_STORAGE')
logger.setLevel(logging.DEBUG)

sqlite3.register_adapter(dict, json.dumps)
sqlite3.register_converter('json', json.loads)

TABLE_NAME = "DeferredTask"


class SQLiteTaskStorage(TaskStorage):
    """SQLite-реализация хранилища отложенных задач."""
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self._conn.row_factory = sqlite3.Row
        self._create_table_if_not_exist()

    def add(self, task: DeferredTask):
        self._conn.execute(
            f"insert into {TABLE_NAME} (id, type, kwargs, after, before) values (?, ?, ?, ?, ?)",
            (task.id, task.type, task.kwargs, task.after, task.before)
        )
        self._conn.commit()
        logger.info(f"Add deferred task (id = {task.id})")

    def delete(self, *task_ids: int):
        self._conn.executemany(
            f"delete from {TABLE_NAME} where id = ?",
            ((id_,) for id_ in task_ids)
        )
        self._conn.commit()
        logger.info(f"Delete tasks (ids = {task_ids})")

    def take_suitable_tasks(self, after_timestamp: Datetime) -> list[DeferredTask]:
        task_dicts = self._conn.execute(
            f"select * from {TABLE_NAME} where after <= ?",
            (after_timestamp,)
        ).fetchall()
        tasks = list(map(lambda t: DeferredTask(**t), task_dicts))
        logger.info(f"Get {len(tasks)} tasks performing after {after_timestamp.isoformat()}")
        task_ids = (t.id for t in tasks)
        self.delete(*task_ids)
        logger.info(f"Delete {len(tasks)} tasks (ids = {task_ids})")
        return tasks

    def get_all(self) -> list[DeferredTask]:
        task_dicts = self._conn.execute(
            f"select * from {TABLE_NAME}"
        ).fetchall()
        return list(map(lambda t: DeferredTask(**t), task_dicts))

    def _create_table_if_not_exist(self):
        query = f"""
            create table {TABLE_NAME} (
                id              INTEGER     PRIMARY KEY,
                type            TEXT        NOT NULL,
                kwargs          json        NOT NULL,
                after           timestamp   NOT NULL,
                before          timestamp   NOT NULL
            );
        """
        try:
            self._conn.execute(query)
            self._conn.commit()
            logger.info(f"Table {TABLE_NAME} was created")
        except sqlite3.OperationalError:
            logger.info(f"Table {TABLE_NAME} has already exist")
