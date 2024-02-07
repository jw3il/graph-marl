import numpy as np


class Buffer:
    """
    Simple buffer that stores data in a numpy array and expands automatically.
    """

    def __init__(self, capacity: int, shape, dtype) -> None:
        """
        Create a buffer with initial capacity.

        :param capacity: initial capacity of the buffer
        :param shape: shape of individual elements
        :param dtype: data type
        """
        self._data = np.empty((capacity, *shape), dtype=dtype)
        self._capacity = capacity
        self._count = 0

    def insert(self, a):
        """
        Insert an element into the buffer and automatically expand the buffer
        if the max capacity has been reached.

        :param a: the element that is inserted
        """
        if isinstance(a, list):
            for e in a:
                self._insert_element(e)
        else:
            self._insert_element(a)

    def _insert_element(self, elem):
        if self._count == self._capacity:
            self._data = np.concatenate((self._data, self._data), axis=0)
            self._capacity = self._data.shape[0]

        self._data[self._count] = elem
        self._count += 1

    def get(self) -> np.ndarray:
        """
        Get the content of the buffer.

        :return: current content
        """
        return self._data[: self._count]

    def clear(self):
        """
        Clears the buffer.
        """
        self._count = 0

    def mean(self, default=0):
        """
        Get the mean of all values.

        :param default: returned if there are no elements, defaults to 0
        :return: mean of all values or default value
        """
        return self.get().mean() if self._count > 0 else default
