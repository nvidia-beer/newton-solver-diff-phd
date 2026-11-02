# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Swept contact must deform a free face while completing the prescribed motion.

The public fixture has no recorded assets: a narrow probe enters the interior
of a coarse tetrahedral block's top triangle, far from every block vertex.
Independent geometric checks reject surface crossings and inverted elements.
Analytic continuous vertex/face and edge/edge tests live in test_diff_phd_ccd.
"""

import itertools
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from tests.test_diff_phd_solver import DEVICE

DENSITY = 800.0
RADIUS = 0.0005
OPTIONS = {"max_iterations": 2000, "abs_tol": 1.0e-9, "rel_tol": 1.0e-6}


def _box(lower, upper):
    corners = list(itertools.product((0, 1), repeat=3))
    lookup = {point: i for i, point in enumerate(corners)}
    lower, upper = np.asarray(lower), np.asarray(upper)
    vertices = lower + np.asarray(corners) * (upper - lower)
    tets = []
    for axes in itertools.permutations(range(3)):
        point = np.zeros(3, dtype=int)
        tet = [0]
        for axis in axes:
            point[axis] = 1
            tet.append(lookup[tuple(point)])
        if np.linalg.det(vertices[tet[1:]] - vertices[tet[:1]]) < 0:
            tet[1], tet[2] = tet[2], tet[1]
        tets.append(tet)
    return vertices, np.asarray(tets, dtype=np.int32)


def _surface(vertices, tets):
    incident = {}
    for tet in tets:
        for opposite in range(4):
            face = np.delete(tet, opposite)
            a, b, c = vertices[face]
            if np.dot(np.cross(b - a, c - a), vertices[tet[opposite]] - a) > 0:
                face[[1, 2]] = face[[2, 1]]
            incident.setdefault(tuple(sorted(face)), []).append(face)
    return np.asarray([faces[0] for faces in incident.values() if len(faces) == 1], dtype=np.int32)


def _scene(*, pin_tool=True):
    die, die_tets = _box((0.0, 0.0, 0.0), (0.1, 0.1, 0.05))
    tool, tool_tets = _box((0.0325, 0.0625, 0.0535), (0.0375, 0.0675, 0.0835))
    split = len(die)
    rest = np.vstack((die, tool)).astype(np.float32)
    tets = np.vstack((die_tets, tool_tets + split))
    faces_left = _surface(die, die_tets)
    faces_right = _surface(tool, tool_tets) + split
    pinned_die = np.flatnonzero(die[:, 2] == 0)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for point in rest:
        builder.add_particle(wp.vec3(*point), wp.vec3(), mass=0.0, radius=0.0)
    young, poisson = 20000.0, 0.3
    for tet in tets:
        volume = builder.add_tetrahedron(
            *map(int, tet),
            k_mu=young / (2.0 * (1.0 + poisson)),
            k_lambda=young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
            k_damp=0.0,
        )
        for node in tet:
            builder.particle_mass[node] += DENSITY * volume / 4.0
    pinned = np.r_[pinned_die, np.arange(split, len(rest))] if pin_tool else pinned_die
    for node in pinned:
        builder.particle_mass[node] = 0.0
    model = builder.finalize(device=DEVICE)
    return model, rest, tets, split, faces_left, faces_right, pinned_die


def _segment_crosses_triangle(start, end, triangle):
    """Independent Moller-Trumbore test; the fixture has no coplanar overlap."""
    a, b, c = triangle
    direction, edge1, edge2 = end - start, b - a, c - a
    p = np.cross(direction, edge2)
    determinant = edge1 @ p
    if abs(determinant) < 1.0e-16:
        return False
    offset = start - a
    u = (offset @ p) / determinant
    q = np.cross(offset, edge1)
    v = (direction @ q) / determinant
    t = (edge2 @ q) / determinant
    return -1.0e-8 <= u and -1.0e-8 <= v and u + v <= 1.0 + 1.0e-8 and 0.0 <= t <= 1.0


def _contained(points, positions, tets):
    vertices = positions[tets]
    matrices = np.transpose(vertices[:, 1:] - vertices[:, :1], (0, 2, 1))
    inverse = np.linalg.inv(matrices)
    bary = np.einsum("tij,ptj->pti", inverse, points[:, None, :] - vertices[None, :, 0])
    # Include shared internal tet faces so a point cannot disappear between
    # elements. The requested positive contact clearance separates the bodies;
    # tet orientation is checked independently before this union query.
    return np.any(np.all(bary >= -1.0e-8, axis=2) & (bary.sum(axis=2) <= 1.0 + 1.0e-8), axis=1)


class TestDiffPhDSweptContact(unittest.TestCase):
    def _assert_valid_separation(self, scene, positions, target):
        _model, rest, tets, split, faces_left, faces_right, pinned_die = scene
        positions = np.asarray(positions, dtype=np.float64)
        self.assertTrue(np.isfinite(positions).all())
        current = positions[tets]
        initial = rest[tets].astype(np.float64)
        jacobians = np.linalg.det(current[:, 1:] - current[:, :1]) / np.linalg.det(initial[:, 1:] - initial[:, :1])
        self.assertGreater(jacobians.min(), 1.0e-6, "inverted/degenerate solids invalidate containment")
        np.testing.assert_array_equal(positions[split:], target[split:])
        np.testing.assert_array_equal(positions[pinned_die], rest[pinned_die])
        # Probe vertices, triangle centroids and cap samples catch containment;
        # triangle-edge intersections also catch crossings between those points.
        tool = positions[split:]
        cap = np.asarray(
            [
                (x, y, tool[:, 2].min())
                for x in np.linspace(tool[:, 0].min(), tool[:, 0].max(), 3)
                for y in np.linspace(tool[:, 1].min(), tool[:, 1].max(), 3)
            ]
        )
        samples = np.vstack((tool, positions[faces_right].mean(axis=1), cap))
        self.assertFalse(np.any(_contained(samples, positions, tets[:6])), "probe enters die interior")
        self.assertFalse(np.any(_contained(positions[:split], positions, tets[6:])), "die enters probe interior")
        for face_left in positions[faces_left]:
            for face_right in positions[faces_right]:
                if np.any(face_left.max(axis=0) < face_right.min(axis=0)) or np.any(
                    face_right.max(axis=0) < face_left.min(axis=0)
                ):
                    continue
                for first, second in ((face_left, face_right), (face_right, face_left)):
                    for i in range(3):
                        self.assertFalse(
                            _segment_crosses_triangle(first[i], first[(i + 1) % 3], second),
                            "die and probe surface triangles intersect",
                        )

    def test_face_interior_sweep_completes_press_hold_and_release(self):
        for material in ("pd", "neo_hookean"):
            with self.subTest(material=material):
                self._press_hold_release(material)

    def _press_hold_release(self, material):
        scene = _scene()
        model, rest, _tets, split, faces_left, faces_right, _pinned_die = scene
        solver = SolverDiffPhD(
            model,
            density=DENSITY,
            material=material,
            swept_contact=(RADIUS, split, faces_left, faces_right),
            **OPTIONS,
        )
        state, following = model.state(), model.state()
        targets = wp.array(rest, dtype=wp.vec3, device=DEVICE)
        # Both initial and uncorrected final cap positions lie beyond the old
        # 6-radius candidate window. The first step crosses a triangle interior.
        self.assertGreater(float(rest[split:, 2].min() - rest[:split, 2].max()), 6.0 * RADIUS)
        drop = 0.0085
        self.assertGreater(drop - 0.0035, 6.0 * RADIUS)
        displacements = [drop] * 5 + [0.5 * drop, 0.0]
        for step, displacement in enumerate(displacements):
            target = rest.copy()
            target[split:, 2] -= displacement
            targets.assign(target)
            solver.step(state, following, None, None, 0.01, dirichlet_positions=targets)
            solver.check_status()
            q = following.particle_q.numpy()
            self._assert_valid_separation(scene, q, target)
            if step == 0:
                self.assertGreater(np.linalg.norm(q[:split] - rest[:split], axis=1).max(), 0.001)
                self.assertLess(float(q[split:, 2].min()), 0.04501, "the full 5 mm indentation must complete")
            # The fixture deliberately damps physical velocities after each
            # step, as an external scene operation, to assess resting contact.
            following.particle_qd.assign(0.3 * following.particle_qd.numpy())
            state, following = following, state

    def test_separated_swept_mode_preserves_contact_free_solution(self):
        model, rest, _tets, split, faces_left, faces_right, _pinned_die = _scene()
        swept = SolverDiffPhD(model, density=DENSITY, swept_contact=(RADIUS, split, faces_left, faces_right), **OPTIONS)
        reference = SolverDiffPhD(model, density=DENSITY, **OPTIONS)
        state, actual, expected = model.state(), model.state(), model.state()
        target = rest.copy()
        target[split:, 2] += 0.01
        targets = wp.array(target, dtype=wp.vec3, device=DEVICE)
        swept.step(state, actual, None, None, 0.01, dirichlet_positions=targets)
        reference.step(state, expected, None, None, 0.01, dirichlet_positions=targets)
        swept.check_status()
        reference.check_status()
        np.testing.assert_allclose(actual.particle_q.numpy(), expected.particle_q.numpy(), rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(actual.particle_qd.numpy(), expected.particle_qd.numpy(), rtol=0.0, atol=1.0e-5)

    def test_unsupported_contact_combinations_are_rejected(self):
        model, _rest, _tets, split, faces_left, faces_right, _pinned_die = _scene()
        options = (
            {"friction": 0.5},
            {"restitution": 0.5},
            {"frictional_boundary": ("mesh", (RADIUS, split, faces_left, faces_right))},
            {"mesh_contact": (RADIUS, 1000.0, 1000.0, 0.0, 0.0, faces_left, faces_right)},
        )
        for unsupported in options:
            with self.subTest(option=next(iter(unsupported))), self.assertRaises((ValueError, NotImplementedError)):
                SolverDiffPhD(
                    model,
                    density=DENSITY,
                    swept_contact=(RADIUS, split, faces_left, faces_right),
                    **unsupported,
                )

    def test_free_probe_and_tape_are_rejected(self):
        model, _rest, _tets, split, faces_left, faces_right, _pinned_die = _scene(pin_tool=False)
        with self.assertRaises((ValueError, NotImplementedError)):
            SolverDiffPhD(model, density=DENSITY, swept_contact=(RADIUS, split, faces_left, faces_right))
        model, rest, _tets, split, faces_left, faces_right, _pinned_die = _scene()
        solver = SolverDiffPhD(model, density=DENSITY, swept_contact=(RADIUS, split, faces_left, faces_right))
        state, following = model.state(), model.state()
        targets = wp.array(rest, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        with wp.Tape(), self.assertRaises((RuntimeError, NotImplementedError)):
            solver.step(state, following, None, None, 0.01, dirichlet_positions=targets)

    def test_immobile_contact_fails_without_accepting_partial_motion(self):
        model, rest, _tets, split, faces_left, faces_right, _pinned_die = _scene()
        inverse_mass = model.particle_inv_mass.numpy()
        inverse_mass[:split] = 0.0
        inverse_mass[0] = 1.0  # One free bottom corner; the contacted top face stays pinned.
        model.particle_inv_mass.assign(inverse_mass)
        solver = SolverDiffPhD(
            model, density=DENSITY, swept_contact=(RADIUS, split, faces_left, faces_right), **OPTIONS
        )
        state, following = model.state(), model.state()
        velocity = np.zeros_like(rest)
        velocity[0] = (0.1, 0.2, 0.0)
        state.particle_qd.assign(velocity)
        target = rest.copy()
        target[split:, 2] -= 0.0085
        targets = wp.array(target, dtype=wp.vec3, device=DEVICE)
        solver.step(state, following, None, None, 0.01, dirichlet_positions=targets)
        self.assertTrue(int(solver.status.numpy()[0]) & 32, "an immobile collision must report swept-contact failure")
        with self.assertRaises(SolverDiffPhD.ConvergenceError):
            solver.check_status()
        np.testing.assert_array_equal(following.particle_q.numpy(), state.particle_q.numpy())
        np.testing.assert_array_equal(following.particle_qd.numpy(), velocity)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA graphs need a CUDA device")
    def test_captured_sweep_replays_changed_targets_like_direct_steps(self):
        if not wp.is_conditional_graph_supported():
            self.skipTest("conditional graph nodes are unavailable")
        scene = _scene()
        model, rest, _tets, split, faces_left, faces_right, _pinned_die = scene
        options = {"density": DENSITY, "swept_contact": (RADIUS, split, faces_left, faces_right), **OPTIONS}
        direct, captured = SolverDiffPhD(model, **options), SolverDiffPhD(model, **options)
        state, direct_out, captured_out = model.state(), model.state(), model.state()
        targets = wp.array(rest, dtype=wp.vec3, device=DEVICE)
        # Build the inverse and compile conditional loops before capture.
        direct.step(state, direct_out, None, None, 0.01, dirichlet_positions=targets)
        captured.step(state, captured_out, None, None, 0.01, dirichlet_positions=targets)
        direct.check_status()
        captured.check_status()
        with wp.ScopedCapture(device=DEVICE) as capture:
            captured.step(state, captured_out, None, None, 0.01, dirichlet_positions=targets)
        for drop in (0.004, 0.0085):
            target = rest.copy()
            target[split:, 2] -= drop
            targets.assign(target)
            direct.step(state, direct_out, None, None, 0.01, dirichlet_positions=targets)
            wp.capture_launch(capture.graph)
            direct.check_status()
            captured.check_status()
            self._assert_valid_separation(scene, captured_out.particle_q.numpy(), target)
            np.testing.assert_allclose(
                captured_out.particle_q.numpy(), direct_out.particle_q.numpy(), rtol=0.0, atol=1.0e-7
            )
            np.testing.assert_allclose(
                captured_out.particle_qd.numpy(), direct_out.particle_qd.numpy(), rtol=0.0, atol=1.0e-5
            )


if __name__ == "__main__":
    unittest.main()
