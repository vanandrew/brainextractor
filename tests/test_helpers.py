import numpy as np
import pytest

from brainextractor.helpers import (
    bresenham3d,
    closest_integer_point,
    diagonal_dot,
    l2norm,
    l2normarray,
    sphere,
)


class TestL2Norm:
    def test_unit_vector_x(self):
        assert l2norm(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_zero_vector(self):
        assert l2norm(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_3_4_5_triangle(self):
        assert l2norm(np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)

    def test_negative_components(self):
        assert l2norm(np.array([-3.0, -4.0, 0.0])) == pytest.approx(5.0)


class TestL2NormArray:
    def test_multiple_vectors(self):
        arr = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = l2normarray(arr)
        np.testing.assert_allclose(result, [5.0, 1.0, 0.0])

    def test_matches_numpy_norm(self):
        rng = np.random.default_rng(seed=42)
        arr = rng.standard_normal((10, 3))
        np.testing.assert_allclose(l2normarray(arr), np.linalg.norm(arr, axis=1))


class TestClosestIntegerPoint:
    def test_exact_integer(self):
        result = closest_integer_point(np.array([3.0, 4.0, 5.0]))
        np.testing.assert_array_equal(result, [3, 4, 5])
        assert result.dtype == np.int64

    def test_round_up(self):
        result = closest_integer_point(np.array([0.9, 0.9, 0.9]))
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_round_down(self):
        result = closest_integer_point(np.array([0.1, 0.1, 0.1]))
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_mixed_axes(self):
        result = closest_integer_point(np.array([0.2, 0.8, 0.5]))
        # 0.5 ties may resolve either way; just ensure each component is in {floor, ceil}
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] in (0, 1)


class TestBresenham3d:
    def test_endpoints_included(self):
        v0 = np.array([0, 0, 0], dtype=np.int64)
        v1 = np.array([5, 0, 0], dtype=np.int64)
        line = bresenham3d(v0, v1)
        np.testing.assert_array_equal(line[0], v0)
        np.testing.assert_array_equal(line[-1], v1)

    def test_axis_aligned_length(self):
        v0 = np.array([0, 0, 0], dtype=np.int64)
        v1 = np.array([4, 0, 0], dtype=np.int64)
        line = bresenham3d(v0, v1)
        assert line.shape == (5, 3)

    def test_diagonal_line(self):
        v0 = np.array([0, 0, 0], dtype=np.int64)
        v1 = np.array([3, 3, 3], dtype=np.int64)
        line = bresenham3d(v0, v1)
        np.testing.assert_array_equal(line[0], v0)
        np.testing.assert_array_equal(line[-1], v1)
        # all intermediate points should be monotonic on each axis
        diffs = np.diff(line, axis=0)
        assert np.all(diffs >= 0)

    def test_negative_direction(self):
        v0 = np.array([5, 5, 5], dtype=np.int64)
        v1 = np.array([0, 0, 0], dtype=np.int64)
        line = bresenham3d(v0, v1)
        np.testing.assert_array_equal(line[0], v0)
        np.testing.assert_array_equal(line[-1], v1)


class TestDiagonalDot:
    def test_unit_vectors(self):
        a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(diagonal_dot(a, b), [1.0, 1.0])

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(diagonal_dot(a, b), [0.0])

    def test_matches_einsum(self):
        rng = np.random.default_rng(seed=0)
        a = rng.standard_normal((8, 3))
        b = rng.standard_normal((8, 3))
        np.testing.assert_allclose(diagonal_dot(a, b), np.einsum("ij,ij->i", a, b))


class TestSphere:
    def test_center_inside_sphere(self):
        shape = (10, 10, 10)
        center = [5, 5, 5]
        mask = sphere(shape, radius=3, position=center)
        assert mask[5, 5, 5]

    def test_far_point_outside_sphere(self):
        shape = (10, 10, 10)
        center = [5, 5, 5]
        mask = sphere(shape, radius=3, position=center)
        assert not mask[0, 0, 0]

    def test_returns_correct_shape(self):
        shape = (8, 9, 10)
        mask = sphere(shape, radius=2, position=[4, 4, 4])
        assert mask.shape == shape

    def test_volume_approximates_formula(self):
        # voxel count should be roughly (4/3) * pi * r^3
        shape = (40, 40, 40)
        radius = 10
        mask = sphere(shape, radius=radius, position=[20, 20, 20])
        expected = (4.0 / 3.0) * np.pi * radius**3
        np.testing.assert_allclose(mask.sum(), expected, rtol=0.05)
