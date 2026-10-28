"""Discover and run example_*.py entry points, including nested categories."""

import argparse
import ast
import runpy
import sys
from pathlib import Path


def discover():
    """Read entry points and descriptions without importing simulation code."""
    examples = {}
    root = Path(__file__).parent
    for path in sorted(root.rglob("example_*.py")):
        relative = path.relative_to(root).with_suffix("")
        if any(not part.isidentifier() or part.startswith("_") for part in relative.parts[:-1]):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not any(
            isinstance(node, ast.If) and ast.unparse(node.test) in ("__name__ == '__main__'", "'__main__' == __name__")
            for node in tree.body
        ):
            continue
        name = path.stem.removeprefix("example_")
        if not name.isidentifier():
            continue
        module = ".".join(relative.parts)
        if name in examples:
            raise ValueError(f"Duplicate example name {name!r}: {examples[name][2]} and {module}")
        description = (ast.get_docstring(tree) or name).splitlines()[0]
        headless = any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "HEADLESS" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
            for node in tree.body
        )
        examples[name] = (description, headless, module)
    return dict(sorted(examples.items()))


def show(examples, *, file):
    width = max(map(len, examples), default=0)
    for index, (name, (description, _, _)) in enumerate(examples.items(), 1):
        print(f"{index:2}) {name:<{width}}  {description}", file=file)


def choose(examples):
    show(examples, file=sys.stderr)
    names = list(examples)
    while True:
        print(f"Select example (1-{len(names)}, name, Enter for 1, q to quit): ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if not line or line.strip().lower() == "q":
            raise SystemExit(0)
        choice = line.strip() or "1"
        if choice in examples:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Invalid selection.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List discovered examples and exit.")
    parser.add_argument("--select", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("example", nargs="?", help="Example name; omit to choose interactively.")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments passed to the chosen example.")
    args = parser.parse_args()
    try:
        examples = discover()
    except ValueError as error:
        parser.error(str(error))
    if not examples:
        parser.error("No runnable example_*.py files found.")
    if args.list:
        show(examples, file=sys.stdout)
        return
    if args.example is None and not sys.stdin.isatty():
        show(examples, file=sys.stderr)
        parser.error("Select an example by name, or run in a terminal for the menu.")
    name = args.example if args.example is not None else choose(examples)
    if name not in examples:
        parser.error(f"Unknown example {name!r}. Choose: {', '.join(examples)}")
    if args.select:
        # Docker uses this before opening a container or requiring a display.
        print(name, "headless" if examples[name][1] else "viewer")
        return
    package = __package__ or Path(__file__).parent.parent.name + ".examples"
    module = f"{package}.{examples[name][2]}"
    sys.argv = [module, *args.arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
