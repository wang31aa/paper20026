#!/usr/bin/env python3
import unittest
import numpy as np
from run_rths import design


class RthsDesignIdentity(unittest.TestCase):
    def test_printed_columns_and_target(self):
        d = np.array([0., 1., 4., 9., 16., 25.])
        f = np.array([10., 11., 14., 19., 26., 35.])
        x, y = design(d, f)
        v = np.gradient(d)
        expected = np.column_stack([np.ones(len(d)-2), d[1:-1], d[:-2],
                                    v[1:-1], np.abs(d[1:-1]),
                                    np.sign(v[1:-1]), f[:-2]])
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_array_equal(y, f[2:])


if __name__ == "__main__":
    unittest.main()
