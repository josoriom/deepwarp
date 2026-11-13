import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

wait_seconds = 0.1


class ReadError(RuntimeError):
    pass


def split_groups(items, group_size):
    items = list(items)
    return [items[start : start + group_size] for start in range(0, len(items), group_size)]


def stack_group(outputs):
    dataset = np.stack([np.asarray(traces, dtype=np.float32) for traces, _ in outputs])
    labels = np.stack([np.asarray(labels, dtype=np.float32) for _, labels in outputs])
    return dataset, labels


class GroupReader:
    def __init__(self, read_file, options, groups, readers, groups_ready):
        self.read_file = read_file
        self.options = options
        self.groups = groups
        self.ready = queue.Queue(maxsize=groups_ready)
        self.stop = threading.Event()
        self.pool = ThreadPoolExecutor(readers)
        self.worker = threading.Thread(target=self.read_all, daemon=True)

    def read_one(self, item):
        try:
            return self.read_file(item, self.options)
        except Exception as error:
            raise ReadError(f"could not read {item}: {error}") from error

    def hand_over(self, entry):
        while not self.stop.is_set():
            try:
                self.ready.put(entry, timeout=wait_seconds)
                return True
            except queue.Full:
                continue
        return False

    def read_all(self):
        try:
            for group in self.groups:
                if self.stop.is_set():
                    return
                if not self.hand_over(stack_group(list(self.pool.map(self.read_one, group)))):
                    return
            self.hand_over(None)
        except BaseException as error:
            self.hand_over(error)

    def take(self):
        while True:
            try:
                return self.ready.get(timeout=wait_seconds)
            except queue.Empty:
                continue

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.worker.join()
        self.pool.shutdown(wait=True)


def read(read_file, files, options, group_size=32, readers=4, groups_ready=2):
    reader = GroupReader(read_file, options, split_groups(files, group_size), readers, groups_ready)
    reader.worker.start()
    try:
        while True:
            entry = reader.take()
            if entry is None:
                return
            if isinstance(entry, BaseException):
                raise entry
            yield entry
    finally:
        reader.close()
