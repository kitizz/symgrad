"""Adapted from https://stackoverflow.com/a/52794817
Changes:
 - Combine reader/write count and lists
 - Use `with` context instead of try/finally patterns
 - Use notify_all instead of deprecated notifyAll alias
 - Add read() and write() context manager methods
 - Hopefully improved readability of lock promotion logic
 - Docs changed for personal taste
"""

from contextlib import contextmanager
import logging
import threading


class ReadWriteLock:
    """A lock object that allows many simultaneous "read locks", but only one "write lock." """

    def __init__(self, with_promotion=False):
        self._read_ready = threading.Condition(threading.RLock())
        self._promote = with_promotion
        self._readers: list[int] = []  # List of Reader thread IDs
        self._writers: list[int] = []  # List of Writer thread IDs

    def acquire_read(self):
        """Acquire a read lock.

        Enures:
         - Blocks while a thread has acquired the write lock.
        """
        logging.debug("RWL : acquire_read()")
        with self._read_ready:
            while self._writers:
                self._read_ready.wait()
            self._readers.append(threading.get_ident())

    def release_read(self):
        """Release a read lock."""
        logging.debug("RWL : release_read()")
        with self._read_ready:
            self._readers.remove(threading.get_ident())
            if not self._readers:
                self._read_ready.notify_all()

    @contextmanager
    def read(self):
        """Acquire a read lock as `with` statement context manager.

        with rw_lock.read():
            # do something...

        """
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    def acquire_write(self):
        """Acquire a write lock.

        Ensures:
         - Blocks until there are no other acquired read or write locks.
        """
        logging.debug("RWL : acquire_write()")
        # A re-entrant lock lets the same thread re-acquire the lock
        self._read_ready.acquire()
        self._writers.append(threading.get_ident())
        while self._readers:
            # Avoid a lock-up when all reading threads are also trying to write.
            thread_acquired_read = threading.get_ident() in self._readers
            all_readers_trying_to_write = set(self._readers).issubset(set(self._writers))
            if self._promote and thread_acquired_read and all_readers_trying_to_write:
                break

            self._read_ready.wait()

    def release_write(self):
        """Release a write lock."""
        logging.debug("RWL : release_write()")
        self._writers.remove(threading.get_ident())
        self._read_ready.notify_all()
        self._read_ready.release()

    @contextmanager
    def write(self):
        """Acquire a write lock as `with` statement context manager.

        with rw_lock.write():
            # do something...

        """
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
