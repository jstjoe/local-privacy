"""Notebook helpers: environment setup, paths, secrets, a shared session.

Every notebook in `notebooks/` starts with the same bootstrap cell and then:

    from opf_eval import nb
    ws = nb.setup()                 # env summary, device, workspace dirs
    S = nb.load_session()           # dataset / sample / detectors shared across notebooks

`Session` is saved to `<workspace>/session.json`, so 02 picks up what you
chose in 01 and 03 scores what 02 ran — but each notebook also runs on its own
with defaults (see `ensure_run`).

Nothing here is Colab-only: on a laptop the workspace is `eval/` inside the
repo checkout (or `$PII_BENCH_HOME`), and secrets come from the environment
or a `.env` file.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default detector set: fast on a T4, no credentials needed.
DEFAULT_DETECTORS = ("presidio", "gliner", "gliner_nvidia", "opf")
# The richest built-in vocabulary (15 coarse categories annotated).
DEFAULT_DATASET = "pii_masking_200k"


def in_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:  # no `google` package at all
        return False


# ------------------------------------------------------------- workspace


@dataclass(frozen=True)
class Workspace:
    """Where fixtures, runs and the session file live."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def runs(self) -> Path:
        return self.root / "results" / "runs"

    @property
    def session_file(self) -> Path:
        return self.root / "session.json"


def _repo_eval_dir() -> Path | None:
    # eval/src/opf_eval/nb.py -> eval/
    here = Path(__file__).resolve()
    candidate = here.parents[2]
    return candidate if (candidate / "pyproject.toml").exists() else None


def workspace(root: str | Path | None = None) -> Workspace:
    """Resolve the workspace: explicit `root` > $PII_BENCH_HOME > /content/pii-bench
    on Colab > the repo's `eval/` dir > ~/pii-bench."""
    if root is None:
        env = os.environ.get("PII_BENCH_HOME")
        if env:
            root = env
        elif in_colab():
            root = "/content/pii-bench"
        else:
            root = _repo_eval_dir() or Path.home() / "pii-bench"
    ws = Workspace(Path(root))
    ws.data.mkdir(parents=True, exist_ok=True)
    ws.runs.mkdir(parents=True, exist_ok=True)
    return ws


# ----------------------------------------------------------- environment


def device(prefer: str | None = None) -> str:
    """`cuda` > `mps` > `cpu`, unless `prefer` (or $PII_BENCH_DEVICE) says otherwise."""
    prefer = prefer or os.environ.get("PII_BENCH_DEVICE")
    if prefer:
        return prefer
    from .runner import autodetect_device

    return autodetect_device()


def setup(*, root: str | Path | None = None, quiet: bool = False) -> Workspace:
    """Prepare the kernel and print a short environment summary."""
    # Triton has no stable Apple Silicon support and isn't needed here; keep
    # OPF on its vanilla PyTorch MoE path. Must be set before `opf` imports.
    os.environ.setdefault("OPF_MOE_TRITON", "0")
    ws = workspace(root)
    if not quiet:
        dev = device()
        gpu = ""
        if dev == "cuda":
            try:
                import torch

                gpu = f" ({torch.cuda.get_device_name(0)})"
            except Exception:  # noqa: BLE001
                pass
        print(f"environment: {'Colab' if in_colab() else 'local'} · python {sys.version.split()[0]}")
        print(f"device:      {dev}{gpu}")
        print(f"workspace:   {ws.root}")
    return ws


def ensure_spacy_model(name: str = "en_core_web_lg") -> None:
    """Install a spaCy pipeline package if it's missing (needed by Presidio).

    Installs with `--no-deps` so pip doesn't upgrade numpy & co. underneath
    the running kernel — the cause of Colab's "restart session" prompt.
    """
    if importlib.util.find_spec(name) is not None:
        print(f"spaCy model {name}: installed")
        return
    import spacy.cli

    print(f"spaCy model {name}: downloading…")
    spacy.cli.download(name, False, False, "--no-deps", "-q")
    importlib.invalidate_caches()
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(
            f"{name} installed but not importable in this kernel — restart the runtime and re-run"
        )


def pip_install(*packages: str) -> None:
    """Install packages into this kernel's interpreter, quietly."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def load_secrets(*names: str, dotenv: str | Path | None = ".env") -> dict[str, bool]:
    """Copy secrets into `os.environ` from Colab Secrets (key icon in the
    sidebar) or a local `.env` file. Existing env vars win. Returns
    {name: found}. Values are never printed."""
    if dotenv and Path(dotenv).exists():
        for line in Path(dotenv).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            os.environ.setdefault(key, value.strip().strip("'\""))
    if in_colab():
        try:
            from google.colab import userdata  # type: ignore[import-not-found]
        except ImportError:
            userdata = None
        for key in names:
            if key in os.environ or userdata is None:
                continue
            try:
                value = userdata.get(key)
            except Exception:  # noqa: BLE001 — SecretNotFoundError, NotebookAccessError
                value = None
            if value:
                os.environ[key] = value
    return {key: bool(os.environ.get(key)) for key in names}


SKYFLOW_SECRETS = ("SKYFLOW_VAULT_URL", "SKYFLOW_VAULT_ID", "SKYFLOW_BEARER_TOKEN")
TOKEN_VAULT_SECRETS = (
    "SKYFLOW_TOKEN_VAULT_URL",
    "SKYFLOW_TOKEN_VAULT_ID",
    "SKYFLOW_TOKEN_BEARER_TOKEN",
)


# --------------------------------------------------------------- session


@dataclass
class Session:
    """What the notebook series is working on. Saved between notebooks."""

    dataset: str = DEFAULT_DATASET
    n: int | None = 100
    seed: int = 42
    detectors: list[str] = field(default_factory=lambda: list(DEFAULT_DETECTORS))
    run_name: str | None = None
    device: str | None = None
    detector_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    root: str | None = None

    @property
    def ws(self) -> Workspace:
        return workspace(self.root)

    @property
    def fixtures_path(self) -> Path:
        size = "all" if self.n is None else str(self.n)
        return self.ws.data / f"{self.dataset}_{size}_s{self.seed}.jsonl"

    @property
    def run_dir(self) -> Path:
        # Derived from the sample by default, so changing dataset / n / seed
        # gives a fresh run dir instead of mixing results.
        name = self.run_name or self.fixtures_path.stem
        return self.ws.runs / name

    def save(self) -> "Session":
        path = self.ws.session_file
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return self

    def show(self) -> None:
        print(f"dataset:   {self.dataset}  (n={self.n}, seed={self.seed})")
        print(f"detectors: {', '.join(self.detectors)}")
        print(f"fixtures:  {self.fixtures_path}")
        print(f"run dir:   {self.run_dir}")


def session(**kwargs: Any) -> Session:
    """Create, save and return a session. Unspecified fields keep the value
    from the saved session (if any), else the defaults."""
    base = dataclasses.asdict(load_session(root=kwargs.get("root"), quiet=True))
    base.update({k: v for k, v in kwargs.items()})
    if isinstance(base.get("detectors"), tuple):
        base["detectors"] = list(base["detectors"])
    s = Session(**base)
    return s.save()


def load_session(*, root: str | Path | None = None, quiet: bool = False) -> Session:
    """The saved session, or defaults if none has been saved yet."""
    path = workspace(root).session_file
    if path.exists():
        data = json.loads(path.read_text())
        known = {f.name for f in dataclasses.fields(Session)}
        s = Session(**{k: v for k, v in data.items() if k in known})
    else:
        s = Session(root=str(root) if root else None)
    if not quiet:
        s.show()
    return s


def ensure_fixtures(s: Session) -> Path:
    from .fixtures import ensure_fixtures as _ensure

    path, reused = _ensure(s.fixtures_path, s.n, dataset=s.dataset, seed=s.seed)
    print(f"{'reusing' if reused else 'wrote'} fixtures: {path}")
    return path


def ensure_run(s: Session, detectors: Iterable[str] | None = None, **run_kwargs: Any) -> Path:
    """Make sure fixtures exist and every detector has results in the run dir;
    runs only what's missing. Lets any notebook run standalone."""
    from .runner import run

    fx = ensure_fixtures(s)
    wanted = list(detectors) if detectors is not None else list(s.detectors)
    todo = [d for d in wanted if not (s.run_dir / f"raw_{d}.jsonl").exists()]
    if todo:
        run(
            fx, todo, s.run_dir,
            device=s.device or device(),
            detector_options=s.detector_options,
            **run_kwargs,
        )
    else:
        print(f"results present for: {', '.join(wanted)}")
    return s.run_dir


# ---------------------------------------------------------------- display


def show_md(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)


def esc(text: str) -> str:
    """Escape a string for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def md_table(
    rows: Sequence[dict],
    columns: Sequence[str] | None = None,
    *,
    headers: dict[str, str] | None = None,
    floatfmt: str = ".3f",
) -> str:
    """Render a list of dicts as a Markdown table. None renders as `—`."""
    if not rows:
        return "_(no rows)_"
    columns = list(columns or rows[0].keys())
    headers = headers or {}

    def cell(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return format(v, floatfmt)
        return esc(v)

    out = [
        "| " + " | ".join(headers.get(c, c) for c in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for r in rows:
        out.append("| " + " | ".join(cell(r.get(c)) for c in columns) + " |")
    return "\n".join(out)


def show_table(rows: Sequence[dict], columns: Sequence[str] | None = None, **kw: Any) -> None:
    show_md(md_table(rows, columns, **kw))


def token_vault():
    """The Skyflow token-vault client for `label_token` mode, or None if the
    `SKYFLOW_TOKEN_VAULT_*` secrets aren't set."""
    load_secrets(*TOKEN_VAULT_SECRETS, "SKYFLOW_BEARER_TOKEN")
    from opf_api.vault_tokens import TokenVaultClient

    return TokenVaultClient.from_env()
