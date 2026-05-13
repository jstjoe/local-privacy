"""Export the FastAPI OpenAPI spec to JSON and YAML on disk.

Invoke via the `opf-api-export-openapi` console script declared in
`api/pyproject.toml`. Importing `opf_api.main` is safe: FastAPI's
`app.openapi()` does not run the `lifespan` context, so the detector
registry and model weights stay untouched. That keeps this script cheap
enough to run in CI freshness checks and in the Pages deploy job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from opf_api.main import app


def _write_json(path: Path, spec: dict) -> None:
    path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, spec: dict) -> None:
    path.write_text(
        yaml.safe_dump(spec, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="docs/api",
        help="Output directory for openapi.json and openapi.yaml (default: docs/api)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = app.openapi()
    _write_json(out_dir / "openapi.json", spec)
    _write_yaml(out_dir / "openapi.yaml", spec)
    print(f"wrote {out_dir / 'openapi.json'}")
    print(f"wrote {out_dir / 'openapi.yaml'}")


if __name__ == "__main__":
    main()
