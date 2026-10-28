# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared material optimization for native DiffPhD examples."""

import sys
from time import perf_counter


class _Progress:
    """Report completed frames without adding GPU synchronization."""

    def __init__(self, label, total, enabled):
        self.label, self.total, self.enabled = label, total, enabled
        self.completed = 0
        self.stream = sys.stdout
        self.terminal = self.stream.isatty()
        self.width = 0

    def __enter__(self):
        self.start = perf_counter()
        self.last_print = -float("inf")
        if self.enabled and self.terminal:
            print(self.label, file=self.stream, flush=True)
        self._print(force=True, newline=True)
        return self

    def update(self, completed):
        self.completed = completed
        # The final frame callback precedes the material VJP. Complete the
        # bar only after the result download, when leaving the context.
        if completed < self.total:
            self._print()

    def __exit__(self, exc_type, _exc, _traceback):
        if exc_type is None:
            self.completed = self.total
        self._print(force=True, newline=True, failed=exc_type is not None)

    def _print(self, *, force=False, newline=False, failed=False):
        if not self.enabled:
            return
        now = perf_counter()
        if not force and now - self.last_print < (0.25 if self.terminal else 5.0):
            return
        elapsed = now - self.start
        fraction = self.completed / self.total
        filled = int(20 * fraction)
        eta = f"~{elapsed * (self.total - self.completed) / self.completed:.0f}s" if self.completed else "--"
        status = "failed" if failed else f"ETA {eta}"
        prefix = "" if self.terminal else f"{self.label} "
        line = (
            f"{prefix}[{'#' * filled}{'-' * (20 - filled)}] "
            f"{self.completed}/{self.total} frames {fraction:3.0%} | {elapsed:.0f}s elapsed | {status}"
        )
        self.width = max(self.width, len(line))
        print(
            "\r" + line.ljust(self.width) if self.terminal else line,
            end="\n" if newline or not self.terminal else "",
            file=self.stream,
            flush=True,
        )
        self.last_print = now
