"""Shared, deterministic demonstration controls for recording and Nsight runs."""


def before_frame(name, example, frame):
    # Match the viewer's Motion selector: finish the released axial benchmark,
    # then start a fresh twist release on the same model. No physical step skips.
    if name == "softbody_diff_phd_cantilever" and frame == 25 and example.simulation.motion == "axial":
        example.test_final()
        example.simulation.set_motion("twist")
        example.sim_time = 0.0
        example.minimum_jacobian = float("inf")
        example.maximum_motion = 0.0
        example.tip_angle = 30.0
        # The unstretched beam is shorter; frame the rotating cross-section
        # closely enough to see it in the README's small animated preview.
        import warp as wp

        example.viewer.set_camera(wp.vec3(0.355, -0.311, 0.209), pitch=-24.0, yaw=122.0)
        return True
    return False


def frame_label(name, example):
    if name == "softbody_diff_phd_cantilever":
        sim = example.simulation
        if sim.motion == "axial":
            return f"Axial force ramp | step {sim.frame}/25 | t = {example.sim_time:.3f} s | slow motion"
        return f"Twist release | tip {example.tip_angle:+.1f} deg | t = {example.sim_time:.3f} s | slow motion"
    if name == "softbody_diff_pd_duck":
        return f"Live forward simulation | t = {example.sim_time:.3f} s | slow-motion playback"
    if name == "softbody_diff_phd_ur5_poke":
        return "White: simulation | blue: recording | experimental: penetration unresolved"
    return "Simulation playback; playback rate is not solver throughput"


def segment_metadata(name, example):
    if name == "softbody_diff_phd_cantilever":
        return {
            "motion": example.simulation.motion,
            "steps": example.simulation.frame,
            "dt": example.simulation.dt,
            "simulated_seconds": example.sim_time,
            "minimum_jacobian": example.minimum_jacobian,
            "maximum_motion_m": example.maximum_motion,
            "final_tip_angle_deg": example.tip_angle,
        }
    if name == "softbody_diff_phd_ur5_poke" and example.forward is not None:
        return {
            "motion": "prescribed poke",
            "steps": example.forward.elapsed_steps,
            "dt": example.forward.dt,
            "simulated_seconds": example.forward.elapsed_steps * example.forward.dt,
            "scene_sha256": example.scene_hash,
            "recording_sha256": example.recording_hash,
            "contact_validated": False,
        }
    return None
