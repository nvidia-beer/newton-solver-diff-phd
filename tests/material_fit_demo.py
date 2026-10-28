# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recover two material stiffnesses from a small synthetic deformation trajectory.

The two connected tetrahedra have fixed Poisson ratio 1/3. Independent log
stiffness scales are optimized with the implicit rollout adjoint. This small
headless check is a synthetic inverse problem, not a reproduction of a paper
experiment or identification from measured data.
"""

import argparse
import json
from contextlib import nullcontext

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD

HEADLESS = True


@wp.kernel
def assign_materials(log_scales: wp.array[float], materials: wp.array2d[float]):
    t = wp.tid()
    scale = wp.exp(log_scales[t])
    materials[t, 0] = 1200.0 * scale
    materials[t, 1] = 2400.0 * scale
    materials[t, 2] = 0.0


@wp.kernel
def trajectory_loss(q: wp.array[wp.vec3], target: wp.array[wp.vec3], loss: wp.array[wp.float64]):
    i = wp.tid()
    error = wp.vec3d(q[i]) - wp.vec3d(target[i])
    wp.atomic_add(loss, 0, wp.float64(10000.0) * wp.dot(error, error))


class MaterialFit:
    """A five-node, two-tetrahedron differentiable calibration fixture."""

    def __init__(self, device="cpu"):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        points = ((0, 0, 0), (0.4, 0, 0), (0, 0.4, 0), (0, 0, 0.4), (0.4, 0.4, 0.4))
        for index, point in enumerate(points):
            builder.add_particle(wp.vec3(*point), wp.vec3(), mass=0.0 if index < 3 else 1.0, radius=0.0)
        builder.add_tetrahedron(0, 1, 2, 3, k_mu=1200.0, k_lambda=2400.0, k_damp=0.0)
        builder.add_tetrahedron(1, 2, 3, 4, k_mu=1200.0, k_lambda=2400.0, k_damp=0.0)
        self.model = builder.finalize(device=device, requires_grad=True)
        self.solver = SolverDiffPhD(
            self.model,
            material="neo_hookean",
            ground_plane=None,
            self_contact=False,
            soft_contact=False,
            tolerance=1e-10,
            max_iterations=500,
        )
        self.initial = self.model.particle_q.numpy().copy()
        self.initial[3] += [0.025, -0.02, 0.05]
        self.initial[4] += [-0.04, 0.03, -0.025]
        self.reference_scales = np.array([0.7, 2.5], dtype=np.float32)
        self.target = self.rollout(np.log(self.reference_scales))[0]

    def rollout(self, log_scales, *, gradient=False):
        # Each call owns all primal states; no overwritten taped intermediates.
        self.model.tet_materials.grad.zero_()
        parameters = wp.array(
            np.asarray(log_scales, dtype=np.float32), device=self.model.device, requires_grad=gradient
        )
        states = [self.model.state(requires_grad=gradient) for _ in range(5)]
        states[0].particle_q.assign(self.initial)
        loss = wp.zeros(1, dtype=wp.float64, device=self.model.device, requires_grad=gradient)
        tape = wp.Tape() if gradient else None
        with tape if tape is not None else nullcontext():
            wp.launch(assign_materials, 2, inputs=[parameters, self.model.tet_materials], device=self.model.device)
            for frame in range(4):
                self.solver.step(states[frame], states[frame + 1], None, None, 0.035)
                if hasattr(self, "target"):
                    wp.launch(
                        trajectory_loss,
                        5,
                        inputs=[states[frame + 1].particle_q, self.target[frame], loss],
                        device=self.model.device,
                    )
        if tape is not None:
            tape.backward(loss)
        outputs = [wp.clone(state.particle_q, requires_grad=False) for state in states[1:]]
        return outputs, float(loss.numpy()[0]), parameters.grad.numpy().copy() if gradient else None


def optimize(device="cpu", iterations=15):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    problem = MaterialFit(device)
    parameters = np.log([1.6, 0.8])
    history = []
    for iteration in range(iterations):
        _, loss, gradient = problem.rollout(parameters, gradient=True)
        history.append(loss)
        print(f"iteration {iteration:02d}: loss={loss:.8g}, stiffness scales={np.exp(parameters)}", flush=True)
        if loss < 1e-9:
            break
        direction = -gradient / max(1.0, float(np.linalg.norm(gradient)))
        accepted = False
        for step_size in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            proposal = parameters + step_size * direction
            _, candidate_loss, _ = problem.rollout(proposal)
            if candidate_loss < loss:
                parameters = proposal
                accepted = True
                break
        if not accepted:
            break
    _, final_loss, _ = problem.rollout(parameters)
    if not np.isfinite(final_loss) or final_loss >= history[0] * 0.1:
        raise RuntimeError("Material optimization did not reduce trajectory loss by at least 90%")
    return {
        "initial_loss": history[0],
        "final_loss": final_loss,
        "stiffness_scales": np.exp(parameters).tolist(),
        "reference_scales": problem.reference_scales.tolist(),
        "iterations": len(history),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Warp device; defaults to CUDA if available")
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(optimize(args.device, args.iterations), indent=2))
