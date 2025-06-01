import traceback
import unittest

import numpy as np

from tklds.utilities import _assert_array_is_2d


class TestUtilities(unittest.TestCase):
    def test_assert_array_is_2d(self):
        _assert_array_is_2d(np.array([[1, 1], [1, 1]]))

    def test_assert_array_is_2d_wrong_shape(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        a_u = arr.reshape(2, 3, 2)
        with self.assertRaises(ValueError):
            try:
                _assert_array_is_2d(a_u)
            except Exception as e:
                error_str = traceback.format_exc()
                raise e
        expected_error = f"Expected 2-d array, received input shape: {a_u.shape}"
        self.assertIn(expected_error, error_str)
