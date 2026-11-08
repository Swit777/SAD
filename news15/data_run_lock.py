"""OS-backed exclusive access to a data release; locks release on process exit."""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path


@contextmanager
def exclusive_data_run(path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0,os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another import is already writing this data release.") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def guard_data_run(path):
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            with exclusive_data_run(path):
                return function(*args,**kwargs)
        return wrapped
    return decorate
