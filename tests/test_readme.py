"""Generated documentation follows actual examples and detects stale output."""

import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


class TestReadme(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = Path(__file__).resolve().parents[1]
        package = tomllib.loads((source / "pyproject.toml").read_text())["project"]["name"].replace("-", "_")
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "newton-solver-check"\ndescription = "A documentation fixture"\ndependencies = []\n'
        )
        (self.root / "tools").mkdir()
        shutil.copyfile(source / "tools/build_readme.py", self.root / "tools/build_readme.py")
        (self.root / "docs").mkdir()
        (self.root / "docs/method.md").write_text("## Method\n\n$$M v = p$$\n")
        self.examples = self.root / "src/newton_solver_check/examples"
        self.examples.mkdir(parents=True)
        shutil.copyfile(source / "src" / package / "examples/__main__.py", self.examples / "__main__.py")
        self.add_example("first")

    def add_example(self, name):
        (self.examples / f"example_{name}.py").write_text(
            f'"""Description for {name}."""\nraise RuntimeError("documentation must not import examples")\n'
            'if __name__ == "__main__":\n    pass\n'
        )

    def build(self, *arguments):
        return subprocess.run(
            [sys.executable, str(self.root / "tools/build_readme.py"), *arguments],
            capture_output=True,
            text=True,
        )

    def test_discovery_and_stale_check(self):
        self.assertEqual(self.build().returncode, 0)
        self.assertEqual(self.build("--check").returncode, 0)
        self.add_example("second")
        stale = self.build("--check")
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("README.md is stale", stale.stderr)
        self.assertEqual(self.build().returncode, 0)
        text = (self.root / "README.md").read_text()
        self.assertIn("Description for second.", text)
        self.assertIn("src/newton_solver_check/examples/example_second.py", text)
        self.assertEqual(text.count("**[second]"), 1)
        self.assertNotIn("Create a solver", text)
        self.assertEqual(self.build("--check").returncode, 0)

    def test_method_edits_and_required_media_are_checked(self):
        self.assertEqual(self.build().returncode, 0)
        (self.root / "docs/method.md").write_text("## Method\n\n$$M a = f$$\n")
        self.assertNotEqual(self.build("--check").returncode, 0)
        required = self.build("--require-videos")
        self.assertNotEqual(required.returncode, 0)
        self.assertIn("Missing video or preview for first", required.stderr)
        self.assertEqual(self.build().returncode, 0)
        self.assertIn("$$M a = f$$", (self.root / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
