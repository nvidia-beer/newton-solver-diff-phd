"""Exercise Docker launch behavior without requiring a daemon or installed solver."""

import json
import os
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

    def invoke(self, script, *arguments, check=True):
        return subprocess.run(
            ["bash", str(self.checkout / "docker" / script), *arguments],
            env=self.environment,
            cwd=self.directory,
            capture_output=True,
            text=True,
            check=check,
        )

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
