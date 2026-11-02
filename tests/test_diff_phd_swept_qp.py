# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent optimality checks for the swept contact's normal response."""

import itertools
import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.swept_qp import (
    IMMOBILE_CONTACT,
    ITERATION_LIMIT,
    CertifiedNormalQP,
)
from tests.test_diff_phd_solver import DEVICE


class TestSweptNormalQP(unittest.TestCase):
    def _solve(self, matrix, gap, *, expected=0, tolerance=1.0e-9, **options):
        matrix = np.asarray(matrix, dtype=np.float64)
        gap = np.asarray(gap, dtype=np.float64)
        size = len(gap)
        qp = CertifiedNormalQP(size, DEVICE, tolerance=tolerance, **options)
        multiplier = wp.zeros(size, dtype=wp.float64, device=DEVICE)
        omega = wp.zeros_like(multiplier)
        qp.solve(
            wp.array(matrix, dtype=wp.float64, device=DEVICE),
            wp.array(gap, dtype=wp.float64, device=DEVICE),
            wp.array(np.ones(size, dtype=np.int32), dtype=int, device=DEVICE),
            multiplier,
            omega,
        )
        self.assertEqual(int(qp.status.numpy()[0]), expected)
        value = multiplier.numpy()
        if expected == 0:
            gradient = matrix @ value + gap
            self.assertGreaterEqual(value.min(), 0.0)
            self.assertLessEqual(np.max(np.abs(np.minimum(gradient, np.diag(matrix) * value))), 1.01 * tolerance)
        else:
            np.testing.assert_array_equal(value, 0.0)
            np.testing.assert_array_equal(omega.numpy(), 0.0)
        return value, qp

    def test_coupled_weighted_rows_match_independent_active_set_enumeration(self):
        rng = np.random.default_rng(20261102)
        for _ in range(12):
            # A scalar inverse panel and weighted 3D contact normals generate
            # the same J A^-1 J.T coupling as the physical response.
            factor = rng.normal(size=(4, 4))
            inverse = factor @ factor.T + np.eye(4)
            coefficients = rng.normal(size=(4, 4))
            normals = rng.normal(size=(4, 3))
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            matrix = (coefficients @ inverse @ coefficients.T) * (normals @ normals.T)
            gap = rng.normal(size=4)
            expected = None
            for subset in itertools.product((False, True), repeat=4):
                active = np.asarray(subset)
                trial = np.zeros(4)
                if active.any():
                    trial[active] = np.linalg.solve(matrix[np.ix_(active, active)], -gap[active])
                gradient = matrix @ trial + gap
                if np.all(trial >= -1.0e-10) and np.all(gradient[~active] >= -1.0e-10):
                    expected = trial
                    break
            self.assertIsNotNone(expected)
            result, _qp = self._solve(matrix, gap)
            np.testing.assert_allclose(result, expected, rtol=1.0e-8, atol=1.0e-10)

    def test_duplicate_constraints_need_no_regularization(self):
        result, _qp = self._solve(np.ones((3, 3)), [-1.0, -1.0, -1.0])
        self.assertAlmostEqual(result.sum(), 1.0, places=12)

    def test_nearly_parallel_rows_exchange_to_independent_constrained_optimum(self):
        # This Gram matrix is strictly positive definite. Pricing first picks
        # row 1, but the constrained optimum needs row 2 instead. Their small
        # independent component gives a 4e-14 normalized Cholesky pivot; treating
        # it as singular stalls coordinate descent above the strict KKT bound.
        factor = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, -1.0e-6, 2.0e-7]])
        matrix = factor @ factor.T
        gap = np.array([-0.001, -0.002, -0.002 + 5.0e-10])
        expected = np.zeros(3)
        active = np.array([0, 2])
        expected[active] = np.linalg.solve(matrix[np.ix_(active, active)], -gap[active])
        self.assertGreater((matrix @ expected + gap)[1], 4.9e-10)

        # Independent diagonal rows force the same exchange beyond the tiled
        # factor's size as well as in the small factorization.
        for padding in (0, 64):
            with self.subTest(padding=padding):
                full_matrix = np.eye(padding + 3)
                full_matrix[padding:, padding:] = matrix
                full_gap = np.r_[np.full(padding, -0.01), gap]
                result, _qp = self._solve(full_matrix, full_gap, tolerance=1.0e-12)
                np.testing.assert_allclose(result, np.r_[np.full(padding, 0.01), expected], rtol=0.0, atol=1.0e-12)

    def test_dependent_entering_row_exchanges_to_stricter_constraint(self):
        # Three constraints in two dimensions: the first priced x-row is
        # superseded by the almost parallel third row after the y-row activates.
        # There is no independent third component and no invertible 3x3 system.
        # Coordinate descent alone stalls near 5e-10, above the requested bound.
        coefficients = np.array(((0.0, 1.0), (1.0, 0.0), (1.0, -1.0e-6)))
        matrix = coefficients @ coefficients.T
        gap = np.array((-0.001, -0.002, -0.002 + 5.0e-10))
        expected = np.zeros(3)
        active = np.array([0, 2])
        expected[active] = np.linalg.solve(matrix[np.ix_(active, active)], -gap[active])
        self.assertGreater((matrix @ expected + gap)[1], 4.9e-10)
        for padding in (0, 64):
            with self.subTest(padding=padding):
                full_matrix = np.eye(padding + 3)
                full_matrix[padding:, padding:] = matrix
                full_gap = np.r_[np.full(padding, -0.01), gap]
                result, qp = self._solve(full_matrix, full_gap, tolerance=1.0e-12)
                np.testing.assert_allclose(result, np.r_[np.full(padding, 0.01), expected], rtol=0.0, atol=1.0e-12)
                # Independently, the closest feasible primal point is the
                # intersection y=.001, x-1e-6*y=.002-5e-10. The discarded dual
                # row x>=.002 remains satisfied with strictly positive slack.
                primal = coefficients.T @ result[padding:]
                np.testing.assert_allclose(primal, [0.002 + 5.0e-10, 0.001], rtol=0.0, atol=1.0e-14)
                self.assertGreaterEqual(float(np.min(coefficients @ primal + gap)), -1.0e-14)
                self.assertEqual(int(qp.fallback.numpy()[0]), 0)
                self.assertLessEqual(int(qp.iterations.numpy()[0]), padding + 8)

    def test_large_candidate_set_preserves_optimum_and_duplicate_tie_order(self):
        # Span several GPU scan strides. Independent diagonal constraints have
        # the analytic nonnegative solution max(0,-q), including the last row.
        size = 259
        gap = np.ones(size)
        gap[[0, 128, 258]] = [-1.0, -2.0, -3.0]
        result, _qp = self._solve(np.eye(size), gap)
        np.testing.assert_allclose(result, np.maximum(0.0, -gap), rtol=0.0, atol=1.0e-12)

        # Redundant equal constraints should choose the same first row as the
        # serial reference scan, rather than a hardware-dependent winner.
        result, _qp = self._solve(np.ones((size, size)), -np.ones(size))
        expected = np.zeros(size)
        expected[0] = 1.0
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1.0e-12)

    def test_incompatible_semidefinite_contacts_fail_and_clear_forces(self):
        _result, qp = self._solve(
            [[1.0, -1.0], [-1.0, 1.0]],
            [-1.0, -1.0],
            expected=ITERATION_LIMIT,
            coordinate_iterations=8,
        )
        self.assertEqual(int(qp.fallback.numpy()[0]), 1)
        self.assertGreater(float(qp.residual.numpy()[0]), 1.0e-9)

    def test_immobile_penetrating_contact_fails(self):
        self._solve([[0.0]], [-0.001], expected=IMMOBILE_CONTACT)

    def test_active_set_budget_is_not_success(self):
        self._solve(np.eye(2), [-1.0, -1.0], expected=ITERATION_LIMIT, max_iterations=1)


if __name__ == "__main__":
    unittest.main()
