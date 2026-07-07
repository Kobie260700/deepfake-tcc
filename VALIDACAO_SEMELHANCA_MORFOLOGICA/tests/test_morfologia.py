import unittest

import numpy as np

from morfologia import align_shape, normalize_shape, shape_rmse


class MorphologyMathTests(unittest.TestCase):
    def test_normalization_removes_translation_and_scale(self):
        shape = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        transformed = shape * 4.0 + np.array([30.0, -7.0])
        self.assertTrue(np.allclose(normalize_shape(shape), normalize_shape(transformed)))

    def test_procrustes_removes_rotation(self):
        shape = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        rotated = shape @ rotation
        self.assertLess(shape_rmse(shape, rotated), 1e-10)

    def test_different_geometry_has_nonzero_distance(self):
        first = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        second = np.array([[0.0, 0.0], [3.0, 0.0], [0.5, 2.0]])
        self.assertGreater(shape_rmse(first, second), 0.01)


if __name__ == "__main__":
    unittest.main()
