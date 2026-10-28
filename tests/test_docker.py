"""Exercise Docker launch behavior without requiring a daemon or installed solver."""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


class TestDocker(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.checkout = self.directory / "renamed checkout with spaces"
        self.checkout.mkdir()
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        source_package = metadata["project"]["name"].replace("-", "_")
        self.package = "newton_solver_launch_check"
        (self.checkout / "pyproject.toml").write_text('[project]\nname = "newton-solver-launch-check"\n')
        (self.checkout / "uv.lock").write_text("version = 1\n")
        shutil.copytree(root / "docker", self.checkout / "docker")
        self.examples = self.checkout / "src" / self.package / "examples"
        self.examples.mkdir(parents=True)
        shutil.copyfile(root / "src" / source_package / "examples/__main__.py", self.examples / "__main__.py")
        for name, headless in (("visual", False), ("terminal", True)):
            (self.examples / f"example_{name}.py").write_text(
                f'"""Discovered {name} example."""\nHEADLESS = {headless}\n'
                'raise RuntimeError("listing must not import simulation code")\n'
                'if __name__ == "__main__":\n    pass\n'
            )
        commands = self.directory / "commands"
        commands.mkdir()
        fake_docker = commands / "docker"
        fake_docker.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_docker.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment["PATH"] = str(commands) + os.pathsep + self.environment["PATH"]
        self.environment.pop("SOLVER_IMAGE", None)
        self.environment.pop("DISPLAY", None)
        self.environment.pop("NSYS_CONFIG_DIRECTIVES", None)
        self.environment.pop("NSYS_ROOT", None)
        self.profiler = self.directory / "nsight install with spaces"
        self.fake_profiler(self.profiler, "2026.5.1.161")
        (self.checkout / "nsys").mkdir()
        (self.checkout / "nsys/profile_examples.py").write_text(
            "import json, sys\nprint(json.dumps({'profile_args': sys.argv[1:]}))\n"
        )

    def invoke(self, script, *arguments, check=True):
        return subprocess.run(
            ["bash", str(self.checkout / "docker" / script), *arguments],
            env=self.environment,
            cwd=self.directory,
            capture_output=True,
            text=True,
            check=check,
        )

    def fake_profiler(self, root, version, *, target_arch=None, exit_code=0):
        architecture = target_arch or ("sbsa-armv8" if platform.machine() == "aarch64" else "x64")
        target = root / f"target-linux-{architecture}" / "nsys"
        target.parent.mkdir(parents=True)
        target.write_text(f'#!/bin/sh\nprintf "NVIDIA Nsight Systems version {version}\\n"\nexit {exit_code}\n')
        target.chmod(0o755)
        (root / "bin").mkdir()
        (root / "bin/nsys").symlink_to(target)
        return root

    def test_build_uses_project_name_and_quoted_context(self):
        argv = json.loads(self.invoke("build.sh", "--platform", "linux/arm64", "--", "--no-cache").stdout)
        self.assertEqual(argv[:3], ["build", "--tag", "newton-solver-launch-check:local"])
        self.assertIn("linux/arm64", argv)
        self.assertIn("--no-cache", argv)
        self.assertEqual(argv[-1], str(self.checkout))

    def test_listing_discovers_new_examples_without_loading_them(self):
        (self.examples / "example_added.py").write_text(
            '"""Discovered later."""\nif __name__ == "__main__":\n    raise RuntimeError("do not run")\n'
        )
        result = self.invoke("run.sh", "--list")
        self.assertIn("added", result.stdout)
        self.assertIn("Discovered later.", result.stdout)
        self.assertIn("visual", result.stdout)
        self.assertNotIn('"run"', result.stdout)

    def test_cpu_headless_launch_is_self_contained(self):
        argv = json.loads(self.invoke("run.sh", "--cpu", "--headless", "visual", "--num-frames", "3").stdout)
        self.assertNotIn("--gpus", argv)
        self.assertIn("CUDA_VISIBLE_DEVICES=", argv)
        mounts = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--mount"]
        self.assertEqual(
            mounts,
            [f"type=bind,src={self.checkout},dst=/workspace", f"type=bind,src={self.checkout}/.cache,dst=/cache"],
        )
        image = argv.index("newton-solver-launch-check:local")
        self.assertEqual(
            argv[image + 1 :],
            [
                "python",
                "-m",
                self.package + ".examples",
                "visual",
                "--device",
                "cpu",
                "--viewer",
                "null",
                "--num-frames",
                "3",
            ],
        )

    def test_terminal_and_explicit_viewer_run_without_display(self):
        for arguments in (("terminal",), ("visual", "--viewer", "null"), ("visual", "--help")):
            with self.subTest(arguments=arguments):
                argv = json.loads(self.invoke("run.sh", *arguments).stdout)
                self.assertIn("--gpus", argv)
                self.assertIn("NVIDIA_DRIVER_CAPABILITIES=compute,utility", argv)
                self.assertNotIn("--hostname", argv)

    def test_raw_command_preserves_arguments(self):
        argv = json.loads(
            self.invoke("run.sh", "--cpu", "--headless", "--image", "custom:tag", "--", "echo", "two words").stdout
        )
        self.assertEqual(argv[-3:], ["custom:tag", "echo", "two words"])

    def test_bad_arguments_and_missing_display_fail_before_docker(self):
        for arguments in (("--image",), ("--",), ("unknown-example",), ("visual",)):
            with self.subTest(arguments=arguments):
                result = self.invoke("run.sh", *arguments, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('"run"', result.stdout)

    def test_nsys_metadata_commands_do_not_need_docker_or_profiler(self):
        self.environment["NSYS_ROOT"] = str(self.directory / "missing")
        for arguments in (("--list",), ("--help",), ("visual", "--dry-run")):
            with self.subTest(arguments=arguments):
                result = self.invoke("run.sh", "--nsys", *arguments)
                self.assertEqual(json.loads(result.stdout), {"profile_args": list(arguments)})
        result = self.invoke("run.sh", "--nsys")
        self.assertEqual(json.loads(result.stdout), {"profile_args": ["--help"]})

    def test_nsys_explicit_override_mounts_host_profiler_and_preserves_arguments(self):
        self.environment["NSYS_ROOT"] = str(self.profiler)
        argv = json.loads(
            self.invoke("run.sh", "--nsys", "visual", "--frames", "5", "--", "--sample-name", "two words").stdout
        )
        self.assertIn(f"type=bind,src={self.profiler},dst=/opt/nsight-systems,readonly", argv)
        self.assertIn("NSYS_PATH=/opt/nsight-systems/bin/nsys", argv)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=compute,utility", argv)
        self.assertNotIn("--privileged", argv)
        self.assertNotIn("--cap-add", argv)
        self.assertNotIn("--security-opt", argv)
        image = argv.index("newton-solver-launch-check:local")
        self.assertEqual(
            argv[image + 1 :],
            [
                "python",
                "/workspace/nsys/profile_examples.py",
                "--device",
                "cuda:0",
                "visual",
                "--frames",
                "5",
                "--",
                "--sample-name",
                "two words",
            ],
        )

    def test_nsys_uses_image_profiler_without_host_discovery(self):
        # A host CLI or user installation must not replace the pinned image CLI.
        self.environment["PATH"] = str(self.profiler / "bin") + os.pathsep + self.environment["PATH"]
        self.environment["NSYS_PATH"] = str(self.profiler / "bin/nsys")
        self.environment["XDG_DATA_HOME"] = str(self.directory / "user data")
        argv = json.loads(self.invoke("run.sh", "--nsys", "visual").stdout)
        mounts = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--mount"]
        self.assertEqual(
            mounts,
            [f"type=bind,src={self.checkout},dst=/workspace", f"type=bind,src={self.checkout}/.cache,dst=/cache"],
        )
        self.assertFalse(any(argument.startswith("NSYS_PATH=") for argument in argv))
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=compute,utility", argv)
        self.assertNotIn("--privileged", argv)

    def test_nsys_rejects_emulated_platform_and_wrong_architecture_override(self):
        wrong_platform = "linux/amd64" if platform.machine() == "aarch64" else "linux/arm64"
        result = self.invoke("run.sh", "--platform", wrong_platform, "--nsys", "visual", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires the host platform", result.stderr)
        wrong_arch = "x64" if platform.machine() == "aarch64" else "sbsa-armv8"
        wrong_profiler = self.fake_profiler(
            self.directory / "wrong architecture", "2026.5.1.161", target_arch=wrong_arch
        )
        self.environment["NSYS_ROOT"] = str(wrong_profiler)
        result = self.invoke("run.sh", "--nsys", "visual", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native target-linux-", result.stderr)
        self.assertNotIn('"run"', result.stdout)

    def test_nsys_forwards_explicit_directives_including_empty_override(self):
        argv = json.loads(self.invoke("run.sh", "--nsys", "visual").stdout)
        self.assertFalse(any(argument.startswith("NSYS_CONFIG_DIRECTIVES=") for argument in argv))
        for directives in ("CuptiDeviceBufferSizeMb=64", ""):
            with self.subTest(directives=directives):
                self.environment["NSYS_CONFIG_DIRECTIVES"] = directives
                argv = json.loads(self.invoke("run.sh", "--nsys", "visual").stdout)
                self.assertIn(f"NSYS_CONFIG_DIRECTIVES={directives}", argv)
                ordinary = json.loads(self.invoke("run.sh", "--cpu", "--headless", "visual").stdout)
                self.assertFalse(any(argument.startswith("NSYS_CONFIG_DIRECTIVES=") for argument in ordinary))

    def test_nsys_validation_and_explicit_gl_fail_before_docker(self):
        for arguments in (("--cpu", "--nsys", "visual"), ("--nsys", "visual", "--viewer", "gl")):
            with self.subTest(arguments=arguments):
                result = self.invoke("run.sh", *arguments, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('"run"', result.stdout)
        self.environment["NSYS_ROOT"] = str(self.directory / "missing")
        result = self.invoke("run.sh", "--nsys", "visual", check=False)
        self.assertIn("NSYS_ROOT", result.stderr)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('"run"', result.stdout)

    def test_nsys_uses_last_viewer_and_ignores_example_arguments_for_docker(self):
        for arguments in (
            ("--viewer", "gl", "--viewer", "null"),
            ("--viewer=gl", "--viewer=null"),
            ("--", "--help", "--viewer", "gl"),
        ):
            with self.subTest(arguments=arguments):
                argv = json.loads(self.invoke("run.sh", "--nsys", "visual", *arguments).stdout)
                self.assertIn("NVIDIA_DRIVER_CAPABILITIES=compute,utility", argv)
                self.assertNotIn("--hostname", argv)

    def test_metadata_mismatch_prevents_command_execution(self):
        fake_cmp = self.directory / "commands" / "cmp"
        fake_cmp.write_text("#!/usr/bin/env bash\nexit 1\n")
        fake_cmp.chmod(0o755)
        result = self.invoke("entrypoint.sh", "echo", "must not execute", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rebuild with ./docker/build.sh", result.stderr)
        self.assertNotIn("must not execute", result.stdout)

    def test_default_container_command_discovers_renamed_package(self):
        commands = self.directory / "commands"
        fake_cmp = commands / "cmp"
        fake_cmp.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_cmp.chmod(0o755)
        fake_python = commands / "python"
        fake_python.write_text(
            "#!/usr/bin/env python3\nimport json, sys\n"
            "if sys.argv[1] == '-c':\n    print('newton-solver-launch-check')\n"
            "else:\n    print(json.dumps(sys.argv[1:]))\n"
        )
        fake_python.chmod(0o755)
        self.assertEqual(
            json.loads(self.invoke("entrypoint.sh").stdout),
            ["-m", "newton_solver_launch_check.examples"],
        )
        self.assertEqual(self.invoke("entrypoint.sh", "echo", "custom command").stdout, "custom command\n")


if __name__ == "__main__":
    unittest.main()
