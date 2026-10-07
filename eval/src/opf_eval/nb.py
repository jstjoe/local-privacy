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

On Colab each notebook gets its own runtime, so by default nothing carries
over between them. Two things make the later notebooks cheap anyway:

- `setup(drive=True)` keeps the workspace in Google Drive, so a run from 02 is
  reused by 03–06 instead of recomputed.
- `ensure_run` fills in results from the baseline shipped with the package
  (see `opf_eval.baseline`) when it matches the sample, so the default
  session can be scored on a CPU-only runtime.

Notebook 07 evaluates detectors on the user's own data, which has no gold
spans. It points a session at an explicit fixtures file instead of a dataset
sample and uses a few extra helpers:

    nb.allow_remote()                       # may hosted LLMs see the data?
                                            # (env PII_BENCH_ALLOW_REMOTE)
    folder = nb.inputs_dir()                # <workspace>/inputs: drop files here
    nb.upload_files()                       # Colab: upload into that folder
    nb.drive_folder("pii-inputs")           # Colab: a folder in My Drive
    S = nb.Session(fixtures=str(fx), detectors=["presidio"])   # fx from fixtures.from_documents
    run_dir = nb.ensure_run(S)              # reruns when the fixtures file changed
    rows = fixtures.map_spans_to_documents(fx, run_dir, "presidio")
    nb.show_spans_in_context(doc.text, rows, doc_id=doc.id)   # label, span, surrounding text
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import hardware
from . import io as _io
from .llm import base as _llm_base

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


# Workspace folders that hold the user's own data or text quoted from it:
# uploaded files, LLM answers, fixtures and silver labels built from those
# files, calibration samples, and runs (raw detector spans and review files
# quote the text they found).
PRIVATE_DIRS = ("inputs", "llm_cache", "data/custom", "data/calibration", "results/runs")
_PRIVATE_GITIGNORE = (
    "# Written by opf_eval.nb: this folder holds your own data, so git ignores all of it.\n*\n"
)


def _in_git_checkout(path: Path) -> bool:
    path = path.resolve()
    return any((p / ".git").exists() for p in (path, *path.parents))


def keep_out_of_git(folder: str | Path) -> Path:
    """Create `folder` with a `.gitignore` that ignores everything in it.

    The `.gitignore` matches itself, so the folder never shows up in
    `git status` and a `git add -A` cannot commit the files in it. An existing
    `.gitignore` is left alone.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / ".gitignore"
    if not marker.exists():
        marker.write_text(_PRIVATE_GITIGNORE, encoding="utf-8")
    return folder


def workspace(root: str | Path | None = None) -> Workspace:
    """Resolve the workspace: explicit `root` > $PII_BENCH_HOME > /content/pii-bench
    on Colab > the repo's `eval/` dir > ~/pii-bench.

    When the workspace sits inside a git checkout (the default on a laptop is
    the repo's own `eval/` folder) the folders in `PRIVATE_DIRS` are created
    with a `.gitignore` that ignores everything in them. Notebook 07 writes
    your files and the PII quoted back by the models there, and this keeps
    them out of every commit.
    """
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
    if _in_git_checkout(ws.root):
        for name in PRIVATE_DIRS:
            keep_out_of_git(ws.root / name)
    return ws


# ----------------------------------------------------------- environment


def device(prefer: str | None = None) -> str:
    """`cuda` > `mps` > `cpu`, unless `prefer` (or $PII_BENCH_DEVICE) says otherwise."""
    prefer = prefer or os.environ.get("PII_BENCH_DEVICE")
    if prefer:
        return prefer
    from .runner import autodetect_device

    return autodetect_device()


DRIVE_MOUNT = "/content/drive"
DRIVE_WORKSPACE = f"{DRIVE_MOUNT}/MyDrive/pii-bench"


def mount_drive() -> Path:
    """Mount Google Drive on Colab and point the workspace at
    `MyDrive/pii-bench`. Asks for permission the first time."""
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(DRIVE_MOUNT)
    os.environ["PII_BENCH_HOME"] = DRIVE_WORKSPACE
    return Path(DRIVE_WORKSPACE)


def setup(
    *, root: str | Path | None = None, drive: bool = False, quiet: bool = False
) -> Workspace:
    """Prepare the kernel and print a short environment summary.

    drive: on Colab, keep the workspace in Google Drive so fixtures, runs and
        the session survive the runtime and are shared by every notebook.
        Ignored outside Colab.
    """
    # Triton has no stable Apple Silicon support and isn't needed here; keep
    # OPF on its vanilla PyTorch MoE path. Must be set before `opf` imports.
    os.environ.setdefault("OPF_MOE_TRITON", "0")
    if drive and root is None:
        if in_colab():
            mount_drive()
        elif not quiet:
            print("drive=True only applies on Colab; using the local workspace")
    ws = workspace(root)
    if not quiet:
        dev = device()
        gpu = ""
        kind, index = _device_kind_index(dev)
        if kind == "cuda":
            try:
                import torch

                gpu = f" ({torch.cuda.get_device_name(index)})"
            except Exception:  # noqa: BLE001
                pass
        env = "Colab" if in_colab() else "local"
        print(f"environment: {env} · python {sys.version.split()[0]}")
        print(f"device:      {dev}{gpu}")
        where = " (Google Drive)" if str(ws.root).startswith(DRIVE_MOUNT) else ""
        print(f"workspace:   {ws.root}{where}")
        if dev == "cpu":
            print(
                "no GPU: detectors run slowly here, "
                "but the default sample is scored from saved results"
            )
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
# Secrets for the LLM labellers and reviewers in notebook 07. Vertex AI needs
# only the project id: credentials come from Google application default
# credentials (see `gcloud_auth`).
# OPENAI_COMPATIBLE_API_KEY is for an `openai_compatible` server that needs a
# key, which `opf_eval.llm` reads from the environment.
LLM_SECRETS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "OPENAI_COMPATIBLE_API_KEY",
)


# ------------------------------------------------------- remote providers

# One source of truth: the env var name, the default and the parsing rule all
# live in `opf_eval.llm.base`, which `make_client` reads too. That module is
# standard library only, so importing it does not need the `llm` extra.
ALLOW_REMOTE_ENV = _llm_base.ALLOW_REMOTE_ENV
ALLOW_REMOTE_DEFAULT = _llm_base.DEFAULT_ALLOW_REMOTE


def allow_remote(flag: bool | None = None) -> bool:
    """Whether hosted providers (Anthropic, OpenAI, Vertex AI…) may see the
    data. Every hosted LLM call in notebook 07 is gated on this.

    `allow_remote(True/False)` sets the switch for this kernel by writing
    `$PII_BENCH_ALLOW_REMOTE`, which `opf_eval.llm.make_client` also reads.
    `allow_remote()` reads it through `opf_eval.llm.remote_allowed`, so the
    notebook and the clients always agree. The values "1", "true", "yes" and
    "on" allow hosted calls. An unset or empty variable means the default,
    which allows them for now. Any other value forbids them so a typo fails
    closed.
    """
    if flag is not None:
        return set_allow_remote(flag)
    return _llm_base.remote_allowed()


_UNSET: Any = object()
# The switch's value before this kernel first overrode it, so that
# `set_allow_remote(None)` can go back to the shell or .env or Secrets value.
_remote_before_override: Any = _UNSET


def set_allow_remote(flag: bool | None) -> bool:
    """Set the hosted-provider switch for this kernel and return the result.

    True and False override `$PII_BENCH_ALLOW_REMOTE` like
    `allow_remote(flag)`. None undoes every override made in this kernel, so
    the switch has the value from your shell or `.env` file or Colab Secrets
    again (as loaded by `load_secrets` before the first override). Without an
    earlier override None changes nothing. Notebook 07's setup cell calls
    this with its `ALLOW_REMOTE` setting on every run, so going from True
    back to None does not keep the old True.
    """
    global _remote_before_override
    if flag is None:
        if _remote_before_override is not _UNSET:
            if _remote_before_override is None:
                os.environ.pop(ALLOW_REMOTE_ENV, None)
            else:
                os.environ[ALLOW_REMOTE_ENV] = _remote_before_override
            _remote_before_override = _UNSET
    else:
        if _remote_before_override is _UNSET:
            _remote_before_override = os.environ.get(ALLOW_REMOTE_ENV)
        os.environ[ALLOW_REMOTE_ENV] = "1" if flag else "0"
    return _llm_base.remote_allowed()


# --------------------------------------------------------------- session


@dataclass
class Session:
    """What the notebook series is working on. Saved between notebooks.

    By default the session works on a sample of a registered dataset
    (`dataset`, `n`, `seed`). Setting `fixtures` points it at an explicit
    fixtures file instead, such as the unlabeled records that
    `fixtures.from_documents` writes for your own files. Then `dataset`, `n`
    and `seed` are ignored, `fixtures_path` is that file and the run dir is
    named after it. A relative `fixtures` path is taken relative to the
    workspace root so a saved session means the same file in every notebook.

    Notebook 07 can build `Session(fixtures=..., detectors=...)` without
    saving it, which leaves the session that notebooks 01–06 share untouched.
    """

    dataset: str = DEFAULT_DATASET
    n: int | None = 100
    seed: int = 42
    detectors: list[str] = field(default_factory=lambda: list(DEFAULT_DETECTORS))
    run_name: str | None = None
    device: str | None = None
    detector_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    root: str | None = None
    fixtures: str | None = None

    @property
    def ws(self) -> Workspace:
        return workspace(self.root)

    @property
    def fixtures_path(self) -> Path:
        if self.fixtures:
            path = Path(self.fixtures).expanduser()
            return path if path.is_absolute() else self.ws.root / path
        size = "all" if self.n is None else str(self.n)
        return self.ws.data / f"{self.dataset}_{size}_s{self.seed}.jsonl"

    @property
    def run_dir(self) -> Path:
        # Derived from the sample (or the explicit fixtures file) by default,
        # so changing dataset / n / seed / fixtures gives a fresh run dir
        # instead of mixing results.
        name = self.run_name or self.fixtures_path.stem
        return self.ws.runs / name

    @property
    def gold(self) -> str | None:
        """The fixtures' gold source from their meta sidecar: "none" for
        unlabeled data, "silver" for LLM silver labels, None when the sidecar
        is missing or says nothing (dataset samples carry real gold)."""
        from .io import read_meta

        return (read_meta(self.fixtures_path) or {}).get("gold")

    def save(self) -> Session:
        path = self.ws.session_file
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return self

    def show(self) -> None:
        if self.fixtures:
            note = {
                _io.GOLD_NONE: "unlabeled: no gold spans",
                _io.GOLD_SILVER: "silver labels from LLMs",
            }
            gold = note.get(self.gold or "", "")
            print(f"data:      your fixtures{f'  ({gold})' if gold else ''}")
        else:
            print(f"dataset:   {self.dataset}  (n={self.n}, seed={self.seed})")
        print(f"detectors: {', '.join(self.detectors)}")
        print(f"fixtures:  {self.fixtures_path}")
        print(f"run dir:   {self.run_dir}")


# Choosing a dataset sample means "stop using the explicit fixtures file".
_SAMPLE_FIELDS = frozenset({"dataset", "n", "seed"})


def session(**kwargs: Any) -> Session:
    """Create, save and return a session. Unspecified fields keep the value
    from the saved session (if any), else the defaults.

    Passing `dataset`, `n` or `seed` without `fixtures` clears a saved
    `fixtures` override, so a notebook that picks a sample gets that sample.
    """
    base = dataclasses.asdict(load_session(root=kwargs.get("root"), quiet=True))
    if _SAMPLE_FIELDS & kwargs.keys() and "fixtures" not in kwargs:
        base["fixtures"] = None
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
        if s.fixtures:
            # Notebooks 01–06 expect a dataset sample with gold spans. A saved
            # override would make them score unlabeled data without saying so.
            print(
                "note: the saved session points at an explicit fixtures file, "
                "not a dataset sample. "
                "Gold-based scores need a sample: call nb.session(dataset=...) to switch back."
            )
    return s


def ensure_fixtures(s: Session) -> Path:
    """The session's fixtures file, materializing the dataset sample if needed.

    With an explicit `s.fixtures` nothing is materialized: the file must
    already exist (write it with `fixtures.from_documents` or
    `fixtures.from_texts`) and is returned as is.
    """
    if s.fixtures:
        path = s.fixtures_path
        if not path.exists():
            raise FileNotFoundError(
                f"fixtures file {path} does not exist; write it first with "
                "fixtures.from_documents(...) or fixtures.from_texts(...)"
            )
        note = {_io.GOLD_NONE: " (unlabeled: no gold spans)", _io.GOLD_SILVER: " (silver labels)"}
        print(f"using fixtures: {path}{note.get(s.gold or '', '')}")
        return path

    from .fixtures import ensure_fixtures as _ensure

    path, reused = _ensure(s.fixtures_path, s.n, dataset=s.dataset, seed=s.seed)
    print(f"{'reusing' if reused else 'wrote'} fixtures: {path}")
    return path


def _clear_if_fixtures_changed(fixtures: Path, run_dir: Path) -> bool:
    """Delete the results in `run_dir` when its manifest records a different
    fixtures hash than `fixtures` has now. Returns True when it cleared them.

    Results without a readable manifest are left alone because there is
    nothing to compare against.
    """
    from .io import file_sha256
    from .runner import _clear_run_dir

    manifest_path = run_dir / "manifest.json"
    try:
        recorded = json.loads(manifest_path.read_text()).get("fixtures_sha256")
    except (OSError, json.JSONDecodeError, AttributeError):
        return False
    if not recorded or recorded == file_sha256(fixtures):
        return False
    _clear_run_dir(run_dir)
    print(f"{fixtures.name} changed since the last run: cleared the old results in {run_dir}")
    return True


def ensure_run(
    s: Session,
    detectors: Iterable[str] | None = None,
    *,
    baseline: bool = True,
    **run_kwargs: Any,
) -> Path:
    """Make sure fixtures exist and every detector has results in the run dir;
    runs only what's missing. Lets any notebook run standalone.

    baseline: fill missing results from the baseline shipped with the package
        when it covers this sample with the same detector options (no GPU
        needed). Pass False to always run the detectors.

    With an explicit `s.fixtures` file the run dir is named after the file,
    so rewriting that file (for example after adding documents and parsing
    again) would leave results for the old contents there. Those results are
    cleared and the detectors run again whenever the file's hash differs
    from the one recorded in the run dir's manifest.

    A raw file that lacks some of the fixtures' records, for example after
    an interrupted run, is removed and that detector runs again.
    """
    from . import baseline as _baseline
    from .runner import run

    fx = ensure_fixtures(s)
    if s.fixtures:
        _clear_if_fixtures_changed(fx, s.run_dir)
    wanted = list(detectors) if detectors is not None else list(s.detectors)
    _drop_incomplete_raw(fx, s.run_dir, wanted)
    todo = [d for d in wanted if not (s.run_dir / f"raw_{d}.jsonl").exists()]
    seeded: list[str] = []
    if todo and baseline:
        seeded = _baseline.seed_run_dir(fx, s.run_dir, todo, detector_options=s.detector_options)
        todo = [d for d in todo if d not in seeded]
        if seeded:
            print(f"loaded saved baseline results for: {', '.join(seeded)}")
    if todo or seeded:
        # With nothing left to run this only writes the manifest.
        run(
            fx, todo, s.run_dir,
            device=s.device or device(),
            detector_options=s.detector_options,
            **run_kwargs,
        )
    else:
        print(f"results present for: {', '.join(wanted)}")
    return s.run_dir


def _drop_incomplete_raw(fixtures: Path, run_dir: Path, detectors: Iterable[str]) -> list[str]:
    """Remove each `raw_<detector>.jsonl` in `run_dir` that has no row for
    some record of `fixtures` and return those detectors. A half-written last
    line counts as missing."""
    want: set[str] | None = None
    dropped: list[str] = []
    for det in detectors:
        path = Path(run_dir) / f"raw_{det}.jsonl"
        if not path.exists():
            continue
        if want is None:
            want = {str(r["id"]) for r in _io.iter_jsonl(fixtures)}
        got: set[str] = set()
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    got.add(str(json.loads(line)["id"]))
                except (ValueError, KeyError, TypeError):
                    continue
        n_have = len(want & got)
        if n_have < len(want):
            print(f"raw_{det}.jsonl covers {n_have} of {len(want)} records (an interrupted run?); "
                  f"running {det} again")
            path.unlink()
            dropped.append(det)
    return dropped


def export_baseline(s: Session, detectors: Iterable[str] | None = None) -> Path:
    """Save this session's run as the package baseline for its dataset and
    seed (in a source checkout: `eval/src/opf_eval/baselines/`). Commit that
    directory to share it."""
    from . import baseline as _baseline

    out = _baseline.export(
        s.fixtures_path, s.run_dir, detectors=list(detectors) if detectors is not None else None
    )
    print(f"baseline written: {out}")
    return out


# ---------------------------------------------------------- your own data


def inputs_dir(*, root: str | Path | None = None) -> Path:
    """`<workspace>/inputs`, created if missing: where notebook 07 looks for
    your files. On a laptop copy files here; on Colab use `upload_files`."""
    path = workspace(root).root / "inputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def upload_files(dest: str | Path | None = None) -> list[Path]:
    """Upload files from your computer into `dest` (default `inputs_dir()`)
    with Colab's file picker. Returns the saved paths in upload order.

    Outside Colab there is no picker, so this raises RuntimeError naming the
    folder to copy files into instead.
    """
    folder = Path(dest) if dest is not None else inputs_dir()
    if not in_colab():
        raise RuntimeError(
            "upload_files() needs Colab's file picker. "
            f"On this machine copy your files into {folder} "
            "and parse that folder instead."
        )
    from google.colab import files  # type: ignore[import-not-found]

    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, data in _colab_upload(files.upload).items():
        # Keep only the base name: the browser supplies it and it must not
        # escape the inputs folder. Uploading a file again replaces it.
        path = folder / Path(name).name
        path.write_bytes(data)
        saved.append(path)
    return saved


def _colab_upload(upload: Any) -> dict[str, bytes]:
    """Call Colab's `files.upload` without leaving copies behind.

    Colab writes every uploaded file to the current directory (`/content`)
    or to `target_dir` before returning the contents, and renames a clash to
    "name (1).ext". Those copies would sit outside the workspace or show up
    as duplicate documents. So the upload goes into a temporary folder that
    is deleted afterwards, and the caller writes the files where they belong.
    """
    import inspect
    import tempfile

    try:
        takes_target = "target_dir" in inspect.signature(upload).parameters
    except (TypeError, ValueError):
        takes_target = False
    with tempfile.TemporaryDirectory(prefix="pii-bench-upload-") as staging:
        if takes_target:
            return upload(target_dir=staging)
        cwd = os.getcwd()
        os.chdir(staging)
        try:
            return upload()
        finally:
            os.chdir(cwd)


DRIVE_ROOT = f"{DRIVE_MOUNT}/MyDrive"


def drive_folder(path: str) -> Path:
    """A folder in your Google Drive (`MyDrive/<path>`), mounting Drive first
    if needed. Only on Colab: on a laptop pass the local folder path instead.

    Unlike `setup(drive=True)` this does not move the workspace to Drive. It
    only reads your files from there.
    """
    if not in_colab():
        raise RuntimeError(
            "drive_folder() mounts Google Drive and only works on Colab. "
            "On this machine pass the local folder path instead."
        )
    if not Path(DRIVE_ROOT).exists():
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(DRIVE_MOUNT)
    p = Path(path)
    # Accept a full /content/drive/... path as well as one relative to MyDrive.
    folder = p if str(p).startswith(DRIVE_MOUNT) else Path(DRIVE_ROOT) / str(p).lstrip("/")
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist in your Google Drive")
    return folder


# ------------------------------------------------------------ cloud & GPU

GCP_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def gcloud_auth() -> bool:
    """Make Google application default credentials available for the Vertex
    AI backends. Returns True when credentials are usable.

    On Colab this opens Colab's sign-in prompt. On a laptop it checks for
    existing credentials and prints how to create them when there are none.
    """
    if in_colab():
        from google.colab import auth  # type: ignore[import-not-found]

        auth.authenticate_user()
        return True
    try:
        import google.auth
    except ImportError:
        print("google-auth is not installed: pip install 'opf-eval[llm]'")
        return False
    try:
        google.auth.default(scopes=list(GCP_SCOPES))
    except Exception as e:  # noqa: BLE001 — DefaultCredentialsError, RefreshError, …
        print(f"no Google application default credentials ({type(e).__name__}).")
        print("run this once in a terminal and then rerun the cell:")
        print("    gcloud auth application-default login")
        return False
    return True


_gcloud_ok = False


def gcloud_ready() -> bool:
    """`gcloud_auth()` until it succeeds once, then True without asking again.

    A failure is not remembered. Rerunning a cell after
    `gcloud auth application-default login` therefore checks again. An
    exception from the sign-in prompt, for example a cancelled prompt on
    Colab, is printed and gives False.
    """
    global _gcloud_ok
    if not _gcloud_ok:
        try:
            _gcloud_ok = bool(gcloud_auth())
        except Exception as e:  # noqa: BLE001 — a cancelled prompt must not crash the cell
            print(f"Google sign-in failed ({type(e).__name__}: {e})")
            return False
    return _gcloud_ok


def warn_if_tracked(notebook: str | Path) -> bool:
    """Print a warning and return True when `notebook` is tracked by git.

    Jupyter saves cell outputs into the notebook file. Notebook 07 quotes
    the personal data it finds, so running the tracked copy in place on your
    own files would put that data into a file that `git add -A` commits. A
    relative path is taken from the current folder, which is the notebook's
    own folder in Jupyter and VS Code.
    """
    path = Path(notebook).resolve()
    if not path.exists() or not _in_git_checkout(path.parent):
        return False
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path.name],
            cwd=path.parent, capture_output=True, text=True, timeout=10, check=False,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    if tracked:
        print(f"warning: {path.name} is tracked by git and Jupyter saves the outputs into it. "
              "The outputs quote the personal data found in your files. Work on a copy outside "
              "the repository, or clear all outputs before you commit.")
    return tracked


def _device_kind_index(dev: str) -> tuple[str, int]:
    """Split a torch device string: "cuda:1" -> ("cuda", 1), "mps" -> ("mps", 0)."""
    kind, _, idx = dev.partition(":")
    try:
        return kind, int(idx) if idx else 0
    except ValueError:
        return kind, 0


def gpu_summary() -> dict[str, Any]:
    """The accelerator this kernel would use and how much memory it has:
    {"device", "name", "vram_gb", "bf16"}.

    `vram_gb` is GiB (1024³ bytes) of device memory on CUDA, or of unified
    system memory on Apple Silicon (MPS), else None. `bf16` says whether the
    device runs bfloat16 natively. Notebook 07 uses this to decide whether
    and how Clef-flash can be loaded. The memory and the bf16 check come from
    `opf_eval.hardware`, which `review.clef.plan_load` uses too, so both
    describe the same card the same way.
    """
    dev = device()
    kind, index = _device_kind_index(dev)
    out: dict[str, Any] = {"device": dev, "name": None, "vram_gb": None, "bf16": False}
    if kind == "cuda":
        try:
            import torch

            out["name"] = torch.cuda.get_device_name(index)
            out["vram_gb"] = round(hardware.cuda_memory_gib(index), 1)
            out["bf16"] = hardware.cuda_native_bf16(index)
        except Exception:  # noqa: BLE001 — broken driver: report what we know
            pass
    elif kind == "mps":
        out["name"] = f"Apple Silicon ({platform.machine() or 'arm64'}, MPS)"
        try:
            mem = hardware.unified_memory_gib()
            out["vram_gb"] = None if mem is None else round(mem, 1)
        except Exception:  # noqa: BLE001
            pass
        # MPS gained bfloat16 in macOS 14; earlier versions raise on use.
        mac = platform.mac_ver()[0]
        try:
            out["bf16"] = int(mac.split(".")[0]) >= 14
        except ValueError:
            out["bf16"] = False
    else:
        out["name"] = platform.processor() or platform.machine() or "cpu"
    return out


# ---------------------------------------------------------------- display


def show_md(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)


_LINE_BREAKS = ("\r\n", "\r", "\n", " ", " ")


def esc(text: str) -> str:
    """Escape a string for a Markdown table cell. Every line break becomes a
    space because CommonMark ends a table row at a lone "\\r" as well as at
    "\\n"."""
    out = str(text).replace("|", "\\|")
    for brk in _LINE_BREAKS:
        out = out.replace(brk, " ")
    return out


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


def _md_inline(text: str) -> str:
    """Escape text for a Markdown table cell, including emphasis markers, so
    that the bold span marker in `spans_in_context` is the only formatting."""
    out = str(text).replace("\\", "\\\\")
    for ch in "*_`[]<>":
        out = out.replace(ch, "\\" + ch)
    return esc(out)


def format_where(where: dict | Sequence[dict] | None) -> str:
    """Render a document location (`documents.Segment.where`) for people:
    {"page": 3} -> "page 3", {"sheet": "People", "cell": "B2"} ->
    "People!B2", {"attachment": "a.pdf", "page": 1} -> "a.pdf › page 1",
    {"content_control": 0, "table": 1, "row": 2, "col": 3} -> "content
    control 0 › table 1 row 2 col 3". A list of locations is joined with
    "; "."""
    if not where:
        return ""
    if not isinstance(where, dict):
        return "; ".join(format_where(w) for w in where if w)
    w = dict(where)
    if "attachment" in w:
        name = w.pop("attachment")
        inner = format_where(w)
        return f"{name} › {inner}" if inner else str(name)
    if "content_control" in w:
        name = f"content control {w.pop('content_control')}"
        inner = format_where(w)
        return f"{name} › {inner}" if inner else name
    if "sheet" in w and "cell" in w:
        return f"{w['sheet']}!{w['cell']}"
    if "part" in w:
        part = str(w.pop("part"))
        if "name" in w:
            part = f"{part} {w.pop('name')}"
        rest = format_where(w)
        return f"{part}, {rest}" if rest else part
    if "table" in w:
        return f"table {w['table']} row {w.get('row')} col {w.get('col')}"
    if "row" in w and "column" in w:
        return f"row {w['row']}, {w['column']}"
    return ", ".join(f"{k} {v}" for k, v in w.items())


def spans_in_context(
    text: str,
    spans: Sequence[dict],
    *,
    width: int = 60,
    limit: int | None = 20,
    doc_id: str | None = None,
) -> str:
    """Markdown table of `spans` with up to `width` characters of `text` on
    each side and the span itself in bold. Spans are shown in text order.

    A span needs `start`/`end` offsets into `text`. A row that names a
    `doc_id` is placed by its `doc_start`/`doc_end` instead, with `text` the
    whole document. Rows from `fixtures.map_spans_to_documents` and from
    `review.llm.disagreements` are such rows. A disagreement row also keeps
    `start`/`end`, which count from the start of its chunk and would put the
    span in the wrong place in the document. Columns for `detector` and
    `where` (page, cell…) appear when any span carries them. Offsets outside
    `text` are skipped. `limit=None` shows every span.

    doc_id: keep only the spans whose `doc_id` is this one. Rows from
        `map_spans_to_documents` cover every document, and their offsets
        only make sense against their own document's text. So when the spans
        name more than one `doc_id` and none is given this raises ValueError
        instead of drawing other documents' spans on `text`.
    """
    if doc_id is not None:
        spans = [s for s in spans if s.get("doc_id") == doc_id]
    else:
        docs = {s.get("doc_id") for s in spans if s.get("doc_id") is not None}
        if len(docs) > 1:
            raise ValueError(
                f"the spans come from {len(docs)} documents; pass doc_id=... to show the "
                "spans of the document whose text this is"
            )

    def bounds(s: dict) -> tuple[int, int] | None:
        if s.get("doc_id") is not None and "doc_start" in s:
            start, end = s.get("doc_start"), s.get("doc_end")
        else:
            start = s.get("start", s.get("doc_start"))
            end = s.get("end", s.get("doc_end"))
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        return (start, end) if 0 <= start < end <= len(text) else None

    placed = sorted(
        ((b, s) for s in spans if (b := bounds(s)) is not None),
        key=lambda item: (item[0][0], item[0][1]),
    )
    if not placed:
        return "_(no spans)_"
    has_det = any(s.get("detector") for _, s in placed)
    has_where = any(s.get("where") for _, s in placed)
    shown = placed if limit is None else placed[:limit]

    columns = (
        ["label", "span"] + (["detector"] if has_det else []) + (["where"] if has_where else [])
    )
    columns.append("context")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for (start, end), s in shown:
        label = str(s.get("label") or s.get("fine_label") or "?")
        fine = s.get("fine_label")
        if fine and fine != label:
            label = f"{label} ({fine})"
        lo, hi = max(0, start - width), min(len(text), end + width)
        before = ("…" if lo > 0 else "") + _md_inline(text[lo:start])
        after = _md_inline(text[end:hi]) + ("…" if hi < len(text) else "")
        piece = text[start:end]
        context = f"{before}{_md_bold(piece, before[-1:], after[:1])}{after}"
        row = [esc(label), _md_inline(piece)]
        if has_det:
            row.append(esc(s.get("detector") or ""))
        if has_where:
            row.append(esc(format_where(s.get("where"))))
        row.append(context)
        lines.append("| " + " | ".join(row) + " |")
    hidden = len(placed) - len(shown)
    if hidden:
        lines.append("")
        lines.append(
            f"_… {hidden} more span{'s' if hidden != 1 else ''} not shown (raise `limit`)._"
        )
    skipped = len(spans) - len(placed)
    if skipped:
        lines.append("")
        lines.append(
            f"_{skipped} span{'s' if skipped != 1 else ''} "
            "with offsets outside the text skipped._"
        )
    return "\n".join(lines)


def _md_bold(piece: str, prev: str = "", nxt: str = "") -> str:
    """`piece` escaped and in bold, given the rendered characters just
    before it (`prev`) and just after it (`nxt`).

    Leading and trailing whitespace stays outside the markers because
    CommonMark does not open or close `**` next to whitespace. CommonMark
    also does not open `**` in front of punctuation that follows a letter
    ("ID**#4521**") or close it after punctuation that a letter follows.
    Those spans are wrapped in `<b>` tags instead, which every notebook
    renderer shows as bold.
    """
    core = piece.strip()
    if not core:
        return _md_inline(piece)
    lead = _md_inline(piece[: len(piece) - len(piece.lstrip())])
    trail = _md_inline(piece[len(piece.rstrip()):])
    body = _md_inline(core)
    before = (prev + lead)[-1:]
    after = (trail + nxt)[:1]
    opens = not _md_punct(body[0]) or not before or before.isspace() or _md_punct(before)
    closes = not _md_punct(body[-1]) or not after or after.isspace() or _md_punct(after)
    if opens and closes:
        return f"{lead}**{body}**{trail}"
    return f"{lead}<b>{body}</b>{trail}"


def _md_punct(ch: str) -> bool:
    """True for what CommonMark counts as punctuation next to a `**` marker."""
    return unicodedata.category(ch)[0] in "PS"


def show_spans_in_context(
    text: str,
    spans: Sequence[dict],
    *,
    width: int = 60,
    limit: int | None = 20,
    doc_id: str | None = None,
) -> None:
    """Display `spans_in_context(text, spans, ...)` as Markdown."""
    show_md(spans_in_context(text, spans, width=width, limit=limit, doc_id=doc_id))


def token_vault():
    """The Skyflow token-vault client for `label_token` mode, or None if the
    `SKYFLOW_TOKEN_VAULT_*` secrets aren't set."""
    load_secrets(*TOKEN_VAULT_SECRETS, "SKYFLOW_BEARER_TOKEN")
    from opf_api.vault_tokens import TokenVaultClient

    return TokenVaultClient.from_env()
