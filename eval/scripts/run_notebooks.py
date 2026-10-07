"""Execute notebooks top to bottom and report which ones fail.

CI uses this as a smoke test for the notebook series:

    uv run python eval/scripts/run_notebooks.py notebooks/03_score_and_compare.ipynb

Each notebook runs in a fresh python3 kernel with its own folder as the
working directory. PII_BENCH_HOME points at a temporary workspace unless it is
already set, so a run never writes into the repo's eval/ folder. The exit code
is non-zero when any notebook fails.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _failing_cell(
    nb: nbformat.NotebookNode,
) -> tuple[int, nbformat.NotebookNode, nbformat.NotebookNode] | None:
    """The first code cell whose outputs hold an error, with its index and that error."""
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                return index, cell, out
    return None


def _report_failure(nb: nbformat.NotebookNode, exc: Exception, timeout: int) -> None:
    if isinstance(exc, CellTimeoutError):
        print(f"  a cell ran longer than the {timeout} s timeout and was interrupted")
    found = _failing_cell(nb)
    if found is None:
        # A timeout or a dead kernel leaves no error output in the notebook
        # so the exception message is all there is to show.
        print("  error:")
        for line in str(exc).splitlines():
            print(f"    {line}")
        return
    index, cell, error = found
    print(f"  failing cell [{index}]:")
    for line in cell.source.splitlines():
        print(f"    | {line}")
    # The kernel colours its traceback with ANSI codes that CI logs show as noise.
    print("  traceback:")
    for frame in error.get("traceback", []):
        for line in ANSI_ESCAPE.sub("", frame).splitlines():
            print(f"    {line}")


def run_notebook(path: Path, *, timeout: int, kernel: str, out_dir: Path | None) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel,
        resources={"metadata": {"path": str(path.parent)}},
    )
    started = time.monotonic()
    error: Exception | None = None
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError, DeadKernelError) as exc:
        error = exc
    elapsed = time.monotonic() - started

    ok = error is None
    print(f"{'PASS' if ok else 'FAIL'} {path} ({elapsed:.1f} s)", flush=True)
    if error is not None:
        _report_failure(nb, error, timeout)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_dir / path.name)
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("notebooks", nargs="+", type=Path, help="notebooks to execute in order")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="seconds each cell may run before it counts as a failure (default 1800)",
    )
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name (default python3)")
    parser.add_argument(
        "--out", type=Path, default=None, help="write executed copies of the notebooks here"
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.notebooks if not p.is_file()]
    if missing:
        parser.error(f"not found: {', '.join(str(p) for p in missing)}")

    tmp: tempfile.TemporaryDirectory[str] | None = None
    if not os.environ.get("PII_BENCH_HOME"):
        tmp = tempfile.TemporaryDirectory(prefix="pii-bench-")
        os.environ["PII_BENCH_HOME"] = tmp.name
    print(f"PII_BENCH_HOME={os.environ['PII_BENCH_HOME']}", flush=True)

    try:
        results = [
            run_notebook(p.resolve(), timeout=args.timeout, kernel=args.kernel, out_dir=args.out)
            for p in args.notebooks
        ]
    finally:
        if tmp is not None:
            tmp.cleanup()

    failed = results.count(False)
    print(f"{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
