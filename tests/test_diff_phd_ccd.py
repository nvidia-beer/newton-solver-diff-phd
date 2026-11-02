# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic continuous vertex/face and edge/edge cases, including missed endpoints."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.ccd import (
    ADVANCEMENT_EXHAUSTED,
    CAPACITY_EXCEEDED,
    INVALID_GEOMETRY,
    SweptMeshCCD,
    _advance,
)


@wp.kernel(enable_backward=False)
def query(
    start: wp.array[wp.vec3d],
    end: wp.array[wp.vec3d],
    kind: int,
    radius: wp.float64,
    maximum: int,
    toi: wp.array[wp.float64],
    coeff: wp.array[wp.vec4d],
    normal: wp.array[wp.vec3d],
    status: wp.array[int],
):
    time, weights, n, result = _advance(start, end, wp.vec4i(0, 1, 2, 3), kind, radius, wp.float64(1.0e-10), maximum)
    toi[0] = time
    coeff[0] = weights
    normal[0] = n
    status[0] = result


class TestSweptMeshCCD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def run_pair(self, start, end, *, kind=0, radius=0.001, maximum=128):
        start = wp.array(np.asarray(start, dtype=np.float64), dtype=wp.vec3d, device=self.device)
        end = wp.array(np.asarray(end, dtype=np.float64), dtype=wp.vec3d, device=self.device)
        toi = wp.zeros(1, dtype=wp.float64, device=self.device)
        coeff = wp.zeros(1, dtype=wp.vec4d, device=self.device)
        normal = wp.zeros(1, dtype=wp.vec3d, device=self.device)
        status = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(query, 1, inputs=[start, end, kind, radius, maximum, toi, coeff, normal, status], device=self.device)
        return float(toi.numpy()[0]), coeff.numpy()[0], normal.numpy()[0], int(status.numpy()[0])

    def test_vertex_crosses_triangle_with_both_endpoints_outside_margin(self):
        start = np.array([[0.2, 0.2, 0.02], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        end = start.copy()
        end[0, 2] = -0.02
        time, coeff, normal, status = self.run_pair(start, end)
        self.assertEqual(status, 1)
        self.assertAlmostEqual(time, 0.475, delta=3.0e-9)
        np.testing.assert_allclose(coeff, [1.0, -0.6, -0.2, -0.2], atol=1.0e-14)
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1.0e-12)

    def test_moving_triangle_crosses_stationary_vertex(self):
        start = np.array([[0.2, 0.2, 0], [0, 0, -0.02], [1, 0, -0.02], [0, 1, -0.02]])
        end = start.copy()
        end[1:, 2] = 0.02
        time, _, normal, status = self.run_pair(start, end)
        self.assertEqual(status, 1)
        self.assertAlmostEqual(time, 0.475, delta=3.0e-9)
        self.assertGreater(normal[2], 0.999999)

    def test_edge_interiors_cross_without_any_vertex_face_hit(self):
        start = np.array([[-1, 0, 0.02], [1, 0, 0.02], [0, -1, 0], [0, 1, 0]])
        end = start.copy()
        end[:2, 2] = -0.02
        time, coeff, normal, status = self.run_pair(start, end, kind=1)
        self.assertEqual(status, 1)
        self.assertAlmostEqual(time, 0.475, delta=3.0e-9)
        np.testing.assert_allclose(coeff, [0.5, 0.5, -0.5, -0.5], atol=1.0e-14)
        np.testing.assert_allclose(normal, [0, 0, 1], atol=1.0e-14)

    def test_parallel_edges_and_point_degenerate_edge(self):
        for first in ([[-1, 0, 0.02], [1, 0, 0.02]], [[0, 0, 0.02], [0, 0, 0.02]]):
            start = np.array([*first, [-1, 0, 0], [1, 0, 0]], dtype=float)
            end = start.copy()
            end[:2, 2] = -0.02
            time, coeff, _, status = self.run_pair(start, end, kind=1)
            self.assertEqual(status, 1)
            self.assertAlmostEqual(time, 0.475, delta=3.0e-9)
            self.assertAlmostEqual(coeff.sum(), 0.0, delta=1.0e-14)

    def test_rigid_common_translation_cancels_from_speed_bound(self):
        start = np.array([[0.2, 0.2, 0.02], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
        end = start + [20.0, -10.0, 3.0]
        _, _, _, status = self.run_pair(start, end)
        self.assertEqual(status, 0)

    def test_resting_contact_has_zero_time(self):
        start = np.array([[0.2, 0.2, 0.001], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
        end = start.copy()
        end[0, 0] += 0.01
        time, _, _, status = self.run_pair(start, end)
        self.assertEqual(status, 1)
        self.assertEqual(time, 0.0)
        _, _, _, physical = self.run_pair(start, end, radius=1.0e-8)
        self.assertEqual(physical, 0)

    def test_search_cap_is_explicit_not_a_miss(self):
        start = np.array([[0.2, 0.2, 0.02], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
        end = start.copy()
        end[0, 2] = -0.02
        _, _, _, status = self.run_pair(start, end, maximum=1)
        self.assertEqual(status, 2)

    def mesh_case(self, *, capacity=64, maximum=128, resting=False):
        start = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 0.001 if resting else 0.02], [2, 0.2, 0.02], [0.2, 2, 0.02]]
        )
        end = start.copy()
        if resting:
            end[3:, 0] += 0.001
        else:
            end[3:, 2] -= 0.04
        detector = SweptMeshCCD([[0, 1, 2]], [[3, 4, 5]], 6, margin=0.001, max_iterations=maximum, device=self.device)
        arrays = [
            wp.zeros(capacity, dtype=dtype, device=self.device) for dtype in (wp.vec4i, wp.vec4d, wp.vec3d, wp.float64)
        ]
        arrays.extend(wp.zeros(1, dtype=int, device=self.device) for _ in range(2))
        args = [
            wp.array(start, dtype=wp.vec3d, device=self.device),
            wp.array(end, dtype=wp.vec3d, device=self.device),
            *arrays,
        ]
        detector.append(*args)
        return detector, args

    def test_full_detector_checks_reverse_direction_and_keeps_rows(self):
        detector, args = self.mesh_case()
        self.assertEqual(int(args[-1].numpy()[0]), 0)
        count = int(args[-2].numpy()[0])
        self.assertGreater(count, 0)
        ids = args[2].numpy()[:count]
        self.assertTrue(np.any(np.all(ids == [3, 0, 1, 2], axis=1)))
        self.assertGreater(int(detector.hit_count.numpy()[0]), 0)
        self.assertLess(float(detector.minimum_toi.numpy()[0]), 0.51)
        saved = [array.numpy().copy() for array in args[2:6]]
        detector.append(*args)
        self.assertEqual(int(args[-2].numpy()[0]), count)
        for before, array in zip(saved, args[2:6], strict=True):
            np.testing.assert_array_equal(before, array.numpy())

    def test_persistent_margin_contact_does_not_block_clear_sweep(self):
        detector, args = self.mesh_case(resting=True)
        self.assertEqual(int(args[-1].numpy()[0]), 0)
        self.assertGreater(int(args[-2].numpy()[0]), 0)
        self.assertEqual(int(detector.hit_count.numpy()[0]), 0)
        self.assertEqual(float(detector.minimum_toi.numpy()[0]), 1.0)

    def test_float32_resting_gap_with_rotating_face_is_conservatively_active(self):
        detector, args = self.mesh_case(resting=True)
        start = args[0].numpy()
        end = start.copy()
        start[3, 2] += 1.0e-9
        end[3, 2] = start[3, 2]
        end[:3, 2] = [-0.0004, 0.0002, 0.0002]
        args[0].assign(start)
        args[1].assign(end)
        args[-2].zero_()
        args[-1].zero_()
        detector.append(*args)
        self.assertEqual(int(args[-1].numpy()[0]), 0)
        self.assertEqual(int(detector.hit_count.numpy()[0]), 0)
        count = int(args[-2].numpy()[0])
        ids = args[2].numpy()[:count]
        row = np.flatnonzero(np.all(ids == [3, 0, 1, 2], axis=1))
        self.assertEqual(len(row), 1)
        self.assertEqual(float(args[5].numpy()[row[0]]), 0.0)

    def test_nonfinite_positions_are_never_reported_clear(self):
        detector, args = self.mesh_case()
        end = args[1].numpy()
        end[0, 0] = np.inf
        args[1].assign(end)
        args[-2].zero_()
        args[-1].zero_()
        detector.append(*args)
        self.assertTrue(int(args[-1].numpy()[0]) & INVALID_GEOMETRY)

    def test_capacity_overflow_is_explicit_and_count_is_bounded(self):
        _, args = self.mesh_case(capacity=1)
        self.assertTrue(int(args[-1].numpy()[0]) & CAPACITY_EXCEEDED)
        self.assertEqual(int(args[-2].numpy()[0]), 1)

    def test_detector_reports_exhausted_narrowphase(self):
        _, args = self.mesh_case(maximum=1)
        self.assertTrue(int(args[-1].numpy()[0]) & ADVANCEMENT_EXHAUSTED)

    def test_proximity_search_cap_retains_witness_when_physical_sweep_is_clear(self):
        # Tangential motion makes the global distance-speed bound very small
        # near the activation margin. The physical gap remains separated by
        # a plane throughout: every right vertex is above every left vertex.
        start = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.2, 0.2, 0.00150002],
                [2, 0.2, 0.00150002],
                [0.2, 2, 0.00150002],
            ],
            dtype=float,
        )
        end = start.copy()
        end[3:, 0] += 0.1
        end[:3, 2] = [-0.0004, 0.0002, 0.0002]
        pair = [3, 0, 1, 2]
        self.assertEqual(self.run_pair(start[pair], end[pair], radius=0.0015)[3], 2)
        self.assertEqual(self.run_pair(start[pair], end[pair], radius=1.0e-8)[3], 0)
        detector = SweptMeshCCD([[0, 1, 2]], [[3, 4, 5]], 6, margin=0.0015, device=self.device)
        arrays = [wp.zeros(64, dtype=dtype, device=self.device) for dtype in (wp.vec4i, wp.vec4d, wp.vec3d, wp.float64)]
        count, flags = (wp.zeros(1, dtype=int, device=self.device) for _ in range(2))
        detector.append(
            wp.array(start, dtype=wp.vec3d, device=self.device),
            wp.array(end, dtype=wp.vec3d, device=self.device),
            *arrays,
            count,
            flags,
        )
        self.assertEqual(int(flags.numpy()[0]), 0)
        self.assertEqual(int(detector.hit_count.numpy()[0]), 0)
        ids = arrays[0].numpy()[: int(count.numpy()[0])]
        row = np.flatnonzero(np.all(ids == pair, axis=1))
        self.assertEqual(len(row), 1)
        # Exhausted searches report the time belonging to their saved geometry,
        # not the next unevaluated advancement point.
        time = float(arrays[3].numpy()[row[0]])
        weights = arrays[1].numpy()[row[0]]
        normal = arrays[2].numpy()[row[0]]
        delta = weights @ (start[pair] + time * (end[pair] - start[pair]))
        np.testing.assert_allclose(delta / np.linalg.norm(delta), normal, atol=1.0e-12)

    def test_invalid_margin_and_shared_topology_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "thickness"):
            SweptMeshCCD([[0, 1, 2]], [[3, 4, 5]], 6, margin=1.0e-9, device=self.device)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            SweptMeshCCD([[0, 1, 2]], [[2, 3, 4]], 5, margin=0.001, device=self.device)


if __name__ == "__main__":
    unittest.main()
