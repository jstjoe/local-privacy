"""Notebook helpers for notebook 07 (evaluate on your own, unlabeled data):
the session's explicit-fixtures override, the remote-provider switch, input
folders, Colab/Drive/GCP helpers (with fake `google.colab` modules), the GPU
summary and the spans-in-context table. No network, no model downloads."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from opf_eval import nb
from opf_eval.detectors import make_span, registry
from opf_eval.io import iter_jsonl, meta_path
from opf_eval.taxonomy import register_vocab


@pytest.fixture
def ws_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PII_BENCH_HOME", str(tmp_path / "ws"))
    return tmp_path / "ws"


def _write_unlabeled(path: Path, texts: dict[str, str], *, gold: str = "none") -> Path:
    """A minimal unlabeled fixtures file + sidecar, shaped like the ones
    `fixtures.from_texts` writes (kept local so this module does not depend
    on that writer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rid, text in texts.items():
            f.write(json.dumps({"id": rid, "text": text, "language": None, "gold_spans": []}) + "\n")
    meta_path(path).write_text(json.dumps({
        "version": 2, "dataset": f"custom:{path.stem}", "source": "texts", "gold": gold,
        "vocab_key": None, "labels": ["EMAIL", "PHONE"], "n_written": len(texts),
    }))
    return path


# ------------------------------------------------- Session.fixtures override


def test_session_defaults_have_no_fixtures_override(ws_root):
    s = nb.Session()
    assert s.fixtures is None
    assert s.fixtures_path.name == f"{nb.DEFAULT_DATASET}_100_s42.jsonl"


def test_explicit_fixtures_sets_paths(ws_root, tmp_path):
    fx = _write_unlabeled(tmp_path / "elsewhere" / "tickets.jsonl", {"t1": "mail a@b.io"})
    s = nb.Session(fixtures=str(fx), detectors=["presidio"])
    assert s.fixtures_path == fx
    # The run dir is named after the fixtures file, not the dataset sample.
    assert s.run_dir == ws_root / "results" / "runs" / "tickets"
    assert nb.Session(fixtures=str(fx), run_name="mine").run_dir == ws_root / "results" / "runs" / "mine"


def test_relative_fixtures_resolve_against_workspace(ws_root):
    s = nb.Session(fixtures="data/custom.jsonl")
    assert s.fixtures_path == ws_root / "data" / "custom.jsonl"
    assert s.run_dir.name == "custom"


def test_ensure_fixtures_returns_explicit_file_without_materializing(ws_root, tmp_path, monkeypatch, capsys):
    fx = _write_unlabeled(tmp_path / "mine.jsonl", {"a": "x"})

    def boom(*a, **k):  # pragma: no cover - would mean we tried to sample a dataset
        raise AssertionError("must not materialize a dataset sample")

    monkeypatch.setattr("opf_eval.fixtures.ensure_fixtures", boom)
    s = nb.Session(fixtures=str(fx))
    assert nb.ensure_fixtures(s) == fx
    out = capsys.readouterr().out
    assert f"using fixtures: {fx}" in out and "unlabeled" in out


def test_ensure_fixtures_missing_explicit_file(ws_root, tmp_path):
    s = nb.Session(fixtures=str(tmp_path / "nope.jsonl"))
    with pytest.raises(FileNotFoundError, match="from_documents"):
        nb.ensure_fixtures(s)


def test_session_gold_from_sidecar(ws_root, tmp_path):
    assert nb.Session(fixtures=str(_write_unlabeled(tmp_path / "u.jsonl", {"a": "x"}))).gold == "none"
    silver = _write_unlabeled(tmp_path / "s.jsonl", {"a": "x"}, gold="silver")
    assert nb.Session(fixtures=str(silver)).gold == "silver"
    # No sidecar at all: nothing to report.
    bare = tmp_path / "bare.jsonl"
    bare.write_text("")
    assert nb.Session(fixtures=str(bare)).gold is None


def test_show_with_fixtures(ws_root, tmp_path, capsys):
    fx = _write_unlabeled(tmp_path / "u.jsonl", {"a": "x"})
    nb.Session(fixtures=str(fx), detectors=["presidio", "gliner"]).show()
    out = capsys.readouterr().out
    assert "unlabeled: no gold spans" in out
    assert f"fixtures:  {fx}" in out
    assert "detectors: presidio, gliner" in out
    assert "dataset:" not in out


def test_session_saves_and_loads_fixtures(ws_root, tmp_path):
    fx = _write_unlabeled(tmp_path / "u.jsonl", {"a": "x"})
    s = nb.session(fixtures=str(fx), detectors=["presidio"])
    loaded = nb.load_session(quiet=True)
    assert loaded == s and loaded.fixtures == str(fx)
    # Changing an unrelated field keeps the override.
    assert nb.session(detectors=["gliner"]).fixtures == str(fx)
    # Picking a dataset sample clears it, so 01–06 get the sample they ask for.
    s3 = nb.session(dataset="openpii_nano")
    assert s3.fixtures is None
    assert s3.fixtures_path.name == "openpii_nano_100_s42.jsonl"


def test_session_sample_change_with_explicit_fixtures_keeps_them(ws_root, tmp_path):
    fx = _write_unlabeled(tmp_path / "u.jsonl", {"a": "x"})
    s = nb.session(n=10, fixtures=str(fx))
    assert s.fixtures == str(fx) and s.n == 10


def test_load_session_flags_saved_fixtures_override(ws_root, tmp_path, capsys):
    fx = _write_unlabeled(tmp_path / "u.jsonl", {"a": "x"})
    nb.session(fixtures=str(fx))
    capsys.readouterr()
    nb.load_session()
    assert "nb.session(dataset=...)" in capsys.readouterr().out
    nb.session(dataset="openpii_nano")
    capsys.readouterr()
    nb.load_session()
    assert "explicit fixtures file" not in capsys.readouterr().out


def test_old_session_file_without_fixtures_field(ws_root):
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "session.json").write_text(json.dumps({"dataset": "openpii_nano", "n": 5, "seed": 1}))
    s = nb.load_session(quiet=True)
    assert s.fixtures is None and s.dataset == "openpii_nano"


# -------------------------------------------- ensure_run on unlabeled data


class _EmailDetector:
    name = "nb07_email"

    def detect(self, text, **_):
        i = text.find("@")
        if i == -1:
            return {"spans": [], "latency_ms": 0.0, "error": None}
        start = text.rfind(" ", 0, i) + 1
        end = text.find(" ", i)
        end = len(text) if end == -1 else end
        return {
            "spans": [make_span("nb07vocab", "mail", start, end, text[start:end])],
            "latency_ms": 0.0,
            "error": None,
        }


register_vocab("nb07vocab", {"mail": "EMAIL"}, kind="detector", overwrite=True)
registry.register_detector(
    registry.DetectorSpec(name="nb07_email", vocab="nb07vocab", factory=lambda ctx: _EmailDetector()),
    overwrite=True,
)


def test_ensure_run_on_unlabeled_fixtures(ws_root, tmp_path, capsys):
    fx = _write_unlabeled(
        ws_root / "data" / "tickets.jsonl",
        {"t1": "write to ann@example.com today", "t2": "no pii here"},
    )
    s = nb.Session(fixtures="data/tickets.jsonl", detectors=["nb07_email"], device="cpu")
    run_dir = nb.ensure_run(s)
    assert run_dir == ws_root / "results" / "runs" / "tickets"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["fixtures"] == str(fx)
    assert manifest["detectors"] == ["nb07_email"]
    # The target labels come from the unlabeled sidecar, not from gold spans.
    assert manifest["labels"] == ["EMAIL", "PHONE"]
    rows = {r["id"]: r for r in iter_jsonl(run_dir / "raw_nb07_email.jsonl")}
    assert [sp["text"] for sp in rows["t1"]["spans"]] == ["ann@example.com"]
    assert rows["t2"]["spans"] == []
    # Second call reuses the results instead of rerunning.
    capsys.readouterr()
    nb.ensure_run(s)
    assert "results present for: nb07_email" in capsys.readouterr().out


def test_ensure_run_reruns_when_fixtures_file_is_rewritten(ws_root, capsys):
    # The notebook-07 loop: parse, run, add documents, parse into the same file, run again.
    _write_unlabeled(ws_root / "data" / "mydocs.jsonl", {"t1": "write to ann@example.com today"})
    s = nb.Session(fixtures="data/mydocs.jsonl", detectors=["nb07_email"], device="cpu")
    run_dir = nb.ensure_run(s)
    _write_unlabeled(
        ws_root / "data" / "mydocs.jsonl",
        {"t1": "please write to bob@corp.org", "t2": "x zed@q.io"},
    )
    capsys.readouterr()
    assert nb.ensure_run(s) == run_dir
    out = capsys.readouterr().out
    assert "results present" not in out
    assert "mydocs.jsonl changed since the last run" in out
    rows = {r["id"]: r for r in iter_jsonl(run_dir / "raw_nb07_email.jsonl")}
    assert sorted(rows) == ["t1", "t2"]
    assert [sp["text"] for sp in rows["t1"]["spans"]] == ["bob@corp.org"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["n_examples"] == 2
    # Same contents again: the fresh results are reused.
    capsys.readouterr()
    nb.ensure_run(s)
    assert "results present for: nb07_email" in capsys.readouterr().out


def test_ensure_run_rewrite_clears_every_detector(ws_root, capsys):
    _write_unlabeled(ws_root / "data" / "d.jsonl", {"t1": "a@b.io"})
    s = nb.Session(fixtures="data/d.jsonl", detectors=["nb07_email"], device="cpu")
    run_dir = nb.ensure_run(s)
    # A stale result of another detector must not survive the rewrite either.
    (run_dir / "raw_other.jsonl").write_text(json.dumps({"id": "t1", "spans": []}) + "\n")
    _write_unlabeled(ws_root / "data" / "d.jsonl", {"t9": "c@d.io"})
    nb.ensure_run(s)
    assert not (run_dir / "raw_other.jsonl").exists()
    assert [r["id"] for r in iter_jsonl(run_dir / "raw_nb07_email.jsonl")] == ["t9"]


def test_ensure_run_reruns_a_detector_whose_raw_file_is_incomplete(ws_root, capsys):
    texts = {f"t{i}": f"mail u{i}@example.com now" for i in range(6)}
    _write_unlabeled(ws_root / "data" / "big.jsonl", texts)
    s = nb.Session(fixtures="data/big.jsonl", detectors=["nb07_email"], device="cpu")
    run_dir = nb.ensure_run(s)
    raw = run_dir / "raw_nb07_email.jsonl"
    # An interrupted run leaves the first rows and maybe half a line.
    lines = raw.read_text().splitlines()
    raw.write_text("\n".join(lines[:3]) + "\n" + lines[3][:10])
    capsys.readouterr()
    nb.ensure_run(s)
    out = capsys.readouterr().out
    assert "raw_nb07_email.jsonl covers 3 of 6 records" in out
    assert "results present" not in out
    assert sorted(r["id"] for r in iter_jsonl(raw)) == sorted(texts)
    # A complete file is reused.
    capsys.readouterr()
    nb.ensure_run(s)
    assert "results present for: nb07_email" in capsys.readouterr().out


# --------------------------------------------------------- allow_remote


def test_allow_remote_default_is_true(monkeypatch):
    monkeypatch.delenv(nb.ALLOW_REMOTE_ENV, raising=False)
    assert nb.allow_remote() is True
    monkeypatch.setenv(nb.ALLOW_REMOTE_ENV, "")
    assert nb.allow_remote() is True


def test_allow_remote_set_writes_env(monkeypatch):
    monkeypatch.delenv(nb.ALLOW_REMOTE_ENV, raising=False)
    assert nb.allow_remote(False) is False
    import os

    assert os.environ[nb.ALLOW_REMOTE_ENV] == "0"
    assert nb.allow_remote() is False
    assert nb.allow_remote(True) is True
    assert os.environ[nb.ALLOW_REMOTE_ENV] == "1"
    assert nb.allow_remote() is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("TRUE", True), (" yes ", True), ("on", True),
     ("0", False), ("false", False), ("no", False), ("off", False), ("ture", False)],
)
def test_allow_remote_parses_env(monkeypatch, value, expected):
    monkeypatch.setenv(nb.ALLOW_REMOTE_ENV, value)
    assert nb.allow_remote() is expected


@pytest.mark.parametrize(
    "value", [None, "", "  ", "1", "true", "TRUE", " yes ", "on", "0", "false", "no", "off", "ture", "2"]
)
def test_allow_remote_matches_the_llm_clients(monkeypatch, value):
    # nb and opf_eval.llm share one env var, one default and one parser.
    from opf_eval import llm

    assert nb.ALLOW_REMOTE_ENV == llm.ALLOW_REMOTE_ENV
    assert nb.ALLOW_REMOTE_DEFAULT == llm.DEFAULT_ALLOW_REMOTE
    if value is None:
        monkeypatch.delenv(nb.ALLOW_REMOTE_ENV, raising=False)
    else:
        monkeypatch.setenv(nb.ALLOW_REMOTE_ENV, value)
    assert nb.allow_remote() is llm.remote_allowed()


def test_set_allow_remote_none_goes_back_to_the_stored_value(monkeypatch, tmp_path):
    # The notebook cell: load the stored value, then apply ALLOW_REMOTE.
    monkeypatch.setattr(nb, "_remote_before_override", nb._UNSET)
    monkeypatch.delenv(nb.ALLOW_REMOTE_ENV, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"{nb.ALLOW_REMOTE_ENV}=0\n")

    def cell(flag):
        nb.load_secrets(nb.ALLOW_REMOTE_ENV, dotenv=env)
        return nb.set_allow_remote(flag)

    assert cell(None) is False
    assert cell(True) is True
    assert cell(None) is False  # .env says 0 again
    assert cell(False) is False
    assert cell(None) is False
    # With nothing stored, None after an override goes back to the default.
    monkeypatch.setattr(nb, "_remote_before_override", nb._UNSET)
    monkeypatch.delenv(nb.ALLOW_REMOTE_ENV, raising=False)
    env.write_text("")
    assert cell(False) is False
    assert cell(None) is True
    import os

    assert nb.ALLOW_REMOTE_ENV not in os.environ


def test_allow_remote_follows_the_llm_default(monkeypatch):
    from opf_eval.llm import base

    monkeypatch.delenv(nb.ALLOW_REMOTE_ENV, raising=False)
    monkeypatch.setattr(base, "DEFAULT_ALLOW_REMOTE", False)
    assert nb.allow_remote() is False


def test_allow_remote_false_switches_off_hosted_clients(monkeypatch):
    from opf_eval import llm

    monkeypatch.delenv(nb.ALLOW_REMOTE_ENV, raising=False)
    nb.allow_remote(False)
    with pytest.raises(llm.LLMError, match="switched off"):
        llm.make_client("anthropic")
    assert llm.make_client("stub").remote is False


def test_nb_imports_without_the_llm_sdks():
    import subprocess

    code = (
        "import sys, opf_eval.nb as nb; nb.allow_remote(); "
        "bad = sorted(m for m in ('anthropic', 'openai') if m in sys.modules); "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_llm_secrets():
    from opf_eval.llm.openai import COMPATIBLE_KEY_ENV

    assert nb.LLM_SECRETS == (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_CLOUD_PROJECT", "OPENAI_COMPATIBLE_API_KEY",
    )
    # Every key the LLM clients read from the environment can come from Colab secrets.
    assert COMPATIBLE_KEY_ENV in nb.LLM_SECRETS


# ------------------------------------------------- inputs / upload / Drive


def _fake_colab(monkeypatch, **submodules: types.ModuleType) -> None:
    pkg = types.ModuleType("google.colab")
    for name, mod in submodules.items():
        setattr(pkg, name, mod)
        monkeypatch.setitem(sys.modules, f"google.colab.{name}", mod)
    monkeypatch.setitem(sys.modules, "google.colab", pkg)
    monkeypatch.setattr(nb, "in_colab", lambda: True)


def test_inputs_dir_created_in_workspace(ws_root):
    d = nb.inputs_dir()
    assert d == ws_root / "inputs" and d.is_dir()
    assert nb.inputs_dir(root=ws_root / "other") == ws_root / "other" / "inputs"


def test_private_folders_are_ignored_by_git_inside_a_checkout(tmp_path, monkeypatch):
    # On a laptop the default workspace is the repo's eval/ folder, so the
    # user's files, fixtures, silver labels and LLM cache must never be committable.
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([git, "init", "-q", str(repo)], check=True)
    monkeypatch.setenv("PII_BENCH_HOME", str(repo / "eval"))
    ws = nb.workspace()
    private = [
        nb.inputs_dir() / "support_ticket.eml",
        ws.root / "llm_cache" / "anthropic" / "m" / "ab" / "x.json",
        ws.data / "custom" / "your_data.jsonl",
        ws.data / "custom" / "your_data.meta.json",
        ws.data / "custom" / "your_data.documents.jsonl",
        ws.data / "custom" / "your_data.silver.jsonl",
        ws.data / "calibration" / "calib_x.jsonl",
        ws.runs / "your_data" / "raw_presidio.jsonl",
        ws.runs / "your_data" / "review_llm_anthropic_claude-opus-5-5.jsonl",
        ws.runs / "your_data" / "review_clef_flash.jsonl",
    ]
    for path in private:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Jane Doe")
    status = subprocess.run(
        [git, "status", "--porcelain", "--untracked-files=all"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert status == ""
    # Calling workspace() again keeps an edited .gitignore as it is.
    marker = ws.root / "inputs" / ".gitignore"
    marker.write_text("*\n!keep.txt\n")
    nb.workspace()
    assert marker.read_text() == "*\n!keep.txt\n"


def test_private_folders_are_not_created_outside_a_checkout(ws_root):
    ws = nb.workspace()
    if any((p / ".git").exists() for p in (ws.root.resolve(), *ws.root.resolve().parents)):
        pytest.skip("the temporary directory is inside a git checkout")
    assert not (ws.root / "llm_cache").exists()
    assert not (nb.inputs_dir() / ".gitignore").exists()


def test_upload_files_outside_colab_names_the_folder(ws_root, monkeypatch):
    monkeypatch.setattr(nb, "in_colab", lambda: False)
    with pytest.raises(RuntimeError, match=str(ws_root / "inputs")):
        nb.upload_files()


def test_upload_files_on_colab(ws_root, monkeypatch):
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: {"notes.txt": b"call 555-0101", "../../escape.csv": b"a,b\n"}
    _fake_colab(monkeypatch, files=files)
    saved = nb.upload_files()
    assert saved == [ws_root / "inputs" / "notes.txt", ws_root / "inputs" / "escape.csv"]
    assert saved[0].read_bytes() == b"call 555-0101"
    # An explicit destination is created on demand.
    dest = ws_root / "inputs" / "batch2"
    assert nb.upload_files(dest) == [dest / "notes.txt", dest / "escape.csv"]


def _colab_like_upload(contents: dict[str, bytes], *, target_dir_arg: bool):
    """Mimics google.colab.files.upload: it saves each file to the current
    directory (or `target_dir`) and renames a clash to "name (1).ext"
    before it returns the contents."""

    def save(where: str) -> dict[str, bytes]:
        for name, data in contents.items():
            path = Path(where or ".") / name
            k = 1
            while path.exists():
                path = path.with_name(f"{Path(name).stem} ({k}){Path(name).suffix}")
                k += 1
            path.write_bytes(data)
        return dict(contents)

    if target_dir_arg:
        return lambda target_dir="": save(target_dir)
    return lambda: save("")


@pytest.mark.parametrize("target_dir_arg", [True, False])
def test_upload_files_leaves_no_copies(ws_root, tmp_path, monkeypatch, target_dir_arg):
    cwd = tmp_path / "content"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    files = types.ModuleType("google.colab.files")
    files.upload = _colab_like_upload({"notes.txt": b"call 555-0101"}, target_dir_arg=target_dir_arg)
    _fake_colab(monkeypatch, files=files)
    assert nb.upload_files() == [ws_root / "inputs" / "notes.txt"]
    # Uploading the same file again replaces it instead of adding "notes (1).txt".
    files.upload = _colab_like_upload({"notes.txt": b"call 555-0199"}, target_dir_arg=target_dir_arg)
    assert nb.upload_files() == [ws_root / "inputs" / "notes.txt"]
    assert sorted(p.name for p in (ws_root / "inputs").iterdir()) == ["notes.txt"]
    assert (ws_root / "inputs" / "notes.txt").read_bytes() == b"call 555-0199"
    # No copy of the upload is left in the working directory.
    assert list(cwd.iterdir()) == []
    assert Path.cwd() == cwd


def test_drive_folder_outside_colab(monkeypatch):
    monkeypatch.setattr(nb, "in_colab", lambda: False)
    with pytest.raises(RuntimeError, match="only works on Colab"):
        nb.drive_folder("pii-inputs")


def test_drive_folder_mounts_once_and_resolves(tmp_path, monkeypatch):
    mount = tmp_path / "drive"
    monkeypatch.setattr(nb, "DRIVE_MOUNT", str(mount))
    monkeypatch.setattr(nb, "DRIVE_ROOT", str(mount / "MyDrive"))
    calls = []

    def fake_mount(where):
        calls.append(where)
        (Path(where) / "MyDrive" / "pii-inputs").mkdir(parents=True)

    drive = types.ModuleType("google.colab.drive")
    drive.mount = fake_mount
    _fake_colab(monkeypatch, drive=drive)

    assert nb.drive_folder("pii-inputs") == mount / "MyDrive" / "pii-inputs"
    assert calls == [str(mount)]
    # Already mounted: no second mount; a full path and a leading slash work too.
    assert nb.drive_folder(str(mount / "MyDrive" / "pii-inputs")) == mount / "MyDrive" / "pii-inputs"
    assert nb.drive_folder("/pii-inputs") == mount / "MyDrive" / "pii-inputs"
    assert calls == [str(mount)]
    with pytest.raises(FileNotFoundError, match="missing"):
        nb.drive_folder("missing")


# ------------------------------------------------------------ gcloud_auth


def test_gcloud_auth_on_colab(monkeypatch):
    called = []
    auth = types.ModuleType("google.colab.auth")
    auth.authenticate_user = lambda: called.append(True)
    _fake_colab(monkeypatch, auth=auth)
    assert nb.gcloud_auth() is True
    assert called == [True]


def test_gcloud_auth_local_with_credentials(monkeypatch):
    google_auth = pytest.importorskip("google.auth")
    seen = {}

    def fake_default(scopes=None, **_):
        seen["scopes"] = scopes
        return object(), "proj"

    monkeypatch.setattr(nb, "in_colab", lambda: False)
    monkeypatch.setattr(google_auth, "default", fake_default)
    assert nb.gcloud_auth() is True
    assert seen["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]


def test_gcloud_auth_local_without_credentials(monkeypatch, capsys):
    google_auth = pytest.importorskip("google.auth")
    from google.auth.exceptions import DefaultCredentialsError

    def fake_default(**_):
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(nb, "in_colab", lambda: False)
    monkeypatch.setattr(google_auth, "default", fake_default)
    assert nb.gcloud_auth() is False
    assert "gcloud auth application-default login" in capsys.readouterr().out


def test_gcloud_auth_without_google_auth_installed(monkeypatch, capsys):
    monkeypatch.setattr(nb, "in_colab", lambda: False)
    monkeypatch.setitem(sys.modules, "google.auth", None)  # makes the import fail
    assert nb.gcloud_auth() is False
    assert "opf-eval[llm]" in capsys.readouterr().out


def test_gcloud_ready_remembers_only_success(monkeypatch, capsys):
    monkeypatch.setattr(nb, "_gcloud_ok", False)
    answers = [False, True]
    calls = []

    def fake_auth():
        calls.append(1)
        return answers.pop(0)

    monkeypatch.setattr(nb, "gcloud_auth", fake_auth)
    assert nb.gcloud_ready() is False
    assert nb.gcloud_ready() is True  # the user signed in and reran the cell
    assert nb.gcloud_ready() is True
    assert len(calls) == 2

    def cancelled():
        raise RuntimeError("user cancelled")

    monkeypatch.setattr(nb, "_gcloud_ok", False)
    monkeypatch.setattr(nb, "gcloud_auth", cancelled)
    assert nb.gcloud_ready() is False
    assert "user cancelled" in capsys.readouterr().out


def test_warn_if_tracked(tmp_path, monkeypatch, capsys):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    subprocess.run([git, "init", "-q", str(repo)], check=True)
    book = repo / "notebooks" / "07.ipynb"
    book.write_text("{}")
    monkeypatch.chdir(repo / "notebooks")
    assert nb.warn_if_tracked("07.ipynb") is False  # not added yet
    subprocess.run([git, "add", "notebooks/07.ipynb"], cwd=repo, check=True)
    assert nb.warn_if_tracked("07.ipynb") is True
    assert "tracked by git" in capsys.readouterr().out
    assert nb.warn_if_tracked("missing.ipynb") is False
    copy = tmp_path / "copy.ipynb"
    copy.write_text("{}")
    assert nb.warn_if_tracked(copy) is False


# ------------------------------------------------------------ gpu_summary


def test_gpu_summary_cpu(monkeypatch):
    monkeypatch.setattr(nb, "device", lambda prefer=None: "cpu")
    out = nb.gpu_summary()
    assert set(out) == {"device", "name", "vram_gb", "bf16"}
    assert out["device"] == "cpu" and out["vram_gb"] is None and out["bf16"] is False
    assert out["name"]


def test_gpu_summary_cuda_with_fake_torch(monkeypatch):
    cuda = types.SimpleNamespace(
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=int(22.5 * 1024**3)),
        get_device_name=lambda i: "NVIDIA L4",
        is_bf16_supported=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(nb, "device", lambda prefer=None: "cuda")
    assert nb.gpu_summary() == {"device": "cuda", "name": "NVIDIA L4", "vram_gb": 22.5, "bf16": True}


def _fake_cuda(monkeypatch, capability, names=("NVIDIA L4",), with_capability=True):
    seen = {}

    def props(i):
        seen.setdefault("props", []).append(i)
        return types.SimpleNamespace(total_memory=int(15 * 1024**3))

    def is_bf16_supported(including_emulation=True):
        # Like torch >= 2.3: emulation counts by default, so a T4 says True.
        return True if including_emulation else capability[0] >= 8

    cuda = types.SimpleNamespace(
        get_device_properties=props,
        get_device_name=lambda i: names[i],
        is_bf16_supported=is_bf16_supported,
    )
    if with_capability:
        cuda.get_device_capability = lambda i: capability
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return seen


@pytest.mark.parametrize("with_capability", [True, False])
def test_gpu_summary_t4_has_no_native_bf16(monkeypatch, with_capability):
    _fake_cuda(monkeypatch, (7, 5), names=("Tesla T4",), with_capability=with_capability)
    monkeypatch.setattr(nb, "device", lambda prefer=None: "cuda")
    out = nb.gpu_summary()
    assert out["name"] == "Tesla T4" and out["vram_gb"] == 15.0
    assert out["bf16"] is False


def test_gpu_summary_indexed_cuda_device(monkeypatch):
    seen = _fake_cuda(monkeypatch, (8, 9), names=("NVIDIA L4", "NVIDIA A100"))
    monkeypatch.setenv("PII_BENCH_DEVICE", "cuda:1")
    out = nb.gpu_summary()
    assert out == {"device": "cuda:1", "name": "NVIDIA A100", "vram_gb": 15.0, "bf16": True}
    assert seen["props"] == [1]


def test_gpu_summary_mps(monkeypatch):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(nb, "device", lambda prefer=None: "mps")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: types.SimpleNamespace(total=24 * 1024**3)
    )
    monkeypatch.setattr(nb.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64"))
    out = nb.gpu_summary()
    assert out["device"] == "mps" and out["vram_gb"] == 24.0 and out["bf16"] is True
    assert "MPS" in out["name"]
    monkeypatch.setattr(nb.platform, "mac_ver", lambda: ("13.6", ("", "", ""), "arm64"))
    assert nb.gpu_summary()["bf16"] is False


# ------------------------------------------------- spans in context


TEXT = "Hi team, please email jane.doe@example.com or call 555-0101 before Friday."


def _span(label, needle, **extra):
    i = TEXT.index(needle)
    return {"label": label, "start": i, "end": i + len(needle), "text": needle, **extra}


def _rows(md: str) -> list[list[str]]:
    lines = [ln for ln in md.splitlines() if ln.startswith("| ")]
    return [[c.strip() for c in ln.strip("|").split(" | ")] for ln in lines]


def test_spans_in_context_basic_table():
    md = nb.spans_in_context(TEXT, [_span("PHONE", "555-0101"), _span("EMAIL", "jane.doe@example.com")], width=10)
    rows = _rows(md)
    assert rows[0] == ["label", "span", "context"]
    # Sorted by position, span in bold, ellipses where the context is cut.
    assert rows[1][:2] == ["EMAIL", "jane.doe@example.com"]
    assert rows[1][2] == "…ase email **jane.doe@example.com** or call 5…"
    assert rows[2][2] == "…m or call **555-0101** before Fr…"


def test_spans_in_context_no_ellipsis_at_text_edges():
    md = nb.spans_in_context("a@b.io", [{"label": "EMAIL", "start": 0, "end": 6}])
    assert _rows(md)[1] == ["EMAIL", "a@b.io", "**a@b.io**"]


def test_spans_in_context_escapes_markdown():
    text = "id|x *bold* user_name `x` <b>"
    i = text.index("user_name")
    md = nb.spans_in_context(text, [{"label": "USERNAME", "start": i, "end": i + 9}], width=100)
    line = md.splitlines()[2]
    assert "user\\_name" in line
    assert "id\\|x \\*bold\\*" in line
    assert "\\`x\\` \\<b\\>" in line
    # Exactly one pair of unescaped bold markers: the span itself.
    assert line.count("**") == 2


def test_spans_in_context_fine_label_detector_and_where_columns():
    spans = [
        _span("EMAIL", "jane.doe@example.com", fine_label="EMAIL", detector="presidio",
              where=[{"sheet": "People", "cell": "B2"}]),
        _span("PHONE", "555-0101", fine_label="PHONE_NUMBER", detector="gliner",
              where={"attachment": "a.pdf", "page": 2}),
    ]
    rows = _rows(nb.spans_in_context(TEXT, spans))
    assert rows[0] == ["label", "span", "detector", "where", "context"]
    assert rows[1][:4] == ["EMAIL", "jane.doe@example.com", "presidio", "People!B2"]
    assert rows[2][:4] == ["PHONE (PHONE_NUMBER)", "555-0101", "gliner", "a.pdf › page 2"]


def test_spans_in_context_accepts_document_offsets():
    i = TEXT.index("555-0101")
    row = {"label": "PHONE", "doc_start": i, "doc_end": i + 8, "where": []}
    rows = _rows(nb.spans_in_context(TEXT, [row], width=5))
    assert rows[1][1] == "555-0101"
    # An empty `where` list does not add a column.
    assert rows[0] == ["label", "span", "context"]


def test_spans_in_context_limit_and_skipped():
    spans = [{"label": "X", "start": i, "end": i + 1} for i in range(5)]
    spans.append({"label": "X", "start": 70, "end": 500})  # outside the text
    md = nb.spans_in_context(TEXT, spans, limit=2)
    assert len(_rows(md)) == 3
    assert "3 more spans not shown" in md
    assert "1 span with offsets outside the text skipped" in md
    assert "not shown" not in nb.spans_in_context(TEXT, spans, limit=None)


def test_spans_in_context_line_breaks_stay_in_one_row():
    text = "Name: Ann\r\nEmail: ann@example.com\rPhone: 555\u2028end"
    i = text.index("ann@")
    md = nb.spans_in_context(text, [{"label": "EMAIL", "start": i, "end": i + 15}], width=100)
    assert "\r" not in md and "\u2028" not in md
    assert md.splitlines()[2] == (
        "| EMAIL | ann@example.com | Name: Ann Email: **ann@example.com** Phone: 555 end |"
    )
    # md_table cells too.
    assert "\r" not in nb.md_table([{"a": "x\ry"}])


def test_spans_in_context_bold_renders_next_to_punctuation():
    markdown_it = pytest.importorskip("markdown_it")
    text = "Ref ID#4521-88 and Mail<jane@x.com> or (Ann) then Bob."
    spans = [
        {"label": label, "start": text.index(v), "end": text.index(v) + len(v)}
        for label, v in [("ID", "#4521-88"), ("EMAIL", "<jane@x.com>"), ("NAME", "Ann"), ("NAME", "Bob.")]
    ]
    md = nb.spans_in_context(text, spans, width=100)
    contexts = [row[-1] for row in _rows(md)[1:]]
    assert "ID<b>#4521-88</b> and" in contexts[0]
    assert "Mail<b>\\<jane@x.com\\></b> or" in contexts[1]
    assert "(**Ann**)" in contexts[2]  # punctuation on both sides of the markers is fine
    assert "then **Bob.**" in contexts[3]
    html = markdown_it.MarkdownIt("commonmark").enable("table").render(md)
    assert "**" not in html
    assert html.count("<strong>") + html.count("<b>") == len(spans)


def test_spans_in_context_whitespace_stays_outside_bold():
    text = "Call John \nnow"
    md = nb.spans_in_context(text, [{"label": "NAME", "start": 5, "end": 10}])
    assert _rows(md)[1] == ["NAME", "John", "Call **John**  now"]
    md = nb.spans_in_context("a  b", [{"label": "X", "start": 1, "end": 3}])
    assert "**" not in md  # nothing but whitespace to mark


def test_spans_in_context_filters_by_doc_id():
    a, b = "Contact ann@example.com now", "Write to zed@q.io or bob@corp.org please"
    rows = [
        {"doc_id": "a", "label": "EMAIL", "doc_start": 8, "doc_end": 23},
        {"doc_id": "b", "label": "EMAIL", "doc_start": 9, "doc_end": 17},
        {"doc_id": "b", "label": "EMAIL", "doc_start": 21, "doc_end": 33},
    ]
    md = nb.spans_in_context(a, rows, doc_id="a")
    assert [r[1] for r in _rows(md)[1:]] == ["ann@example.com"]
    assert "skipped" not in md
    assert [r[1] for r in _rows(nb.spans_in_context(b, rows, doc_id="b"))[1:]] == [
        "zed@q.io", "bob@corp.org",
    ]
    # Rows from several documents without doc_id would draw b's spans on a's text.
    with pytest.raises(ValueError, match="doc_id"):
        nb.spans_in_context(a, rows)
    # Rows of a single document need no doc_id.
    assert "ann@example.com" in nb.spans_in_context(a, rows[:1])
    assert nb.spans_in_context(a, rows, doc_id="missing") == "_(no spans)_"


def test_spans_in_context_places_review_rows_by_document_offsets():
    # review.llm.disagreements rows keep chunk offsets in start/end and add
    # doc_id/doc_start/doc_end. Shown on the whole document's text they must
    # use the document offsets.
    doc = "First chunk text here.\nSecond chunk: mail bob@corp.org today."
    chunk_start = doc.index("Second")
    chunk = doc[chunk_start:]
    i = chunk.index("bob@corp.org")
    row = {
        "id": "d#1", "kind": "span", "label": "EMAIL", "text": "bob@corp.org",
        "start": i, "end": i + 12,
        "doc_id": "d", "doc_start": chunk_start + i, "doc_end": chunk_start + i + 12,
    }
    assert [r[1] for r in _rows(nb.spans_in_context(doc, [row], doc_id="d"))[1:]] == ["bob@corp.org"]
    assert [r[1] for r in _rows(nb.spans_in_context(doc, [row]))[1:]] == ["bob@corp.org"]
    # A row without a doc_id still uses start/end against the chunk text.
    plain = {k: v for k, v in row.items() if not k.startswith("doc_")}
    assert [r[1] for r in _rows(nb.spans_in_context(chunk, [plain]))[1:]] == ["bob@corp.org"]


def test_show_spans_in_context_passes_doc_id(monkeypatch):
    rows = [
        {"doc_id": "a", "label": "EMAIL", "doc_start": 0, "doc_end": 6},
        {"doc_id": "b", "label": "EMAIL", "doc_start": 2, "doc_end": 4},
    ]
    shown = []
    monkeypatch.setattr(nb, "show_md", shown.append)
    nb.show_spans_in_context("a@b.io", rows, doc_id="a")
    assert shown == [nb.spans_in_context("a@b.io", rows, doc_id="a")]


def test_spans_in_context_empty():
    assert nb.spans_in_context(TEXT, []) == "_(no spans)_"


def test_show_spans_in_context_prints_without_ipython(monkeypatch, capsys):
    shown = []
    monkeypatch.setattr(nb, "show_md", shown.append)
    nb.show_spans_in_context(TEXT, [_span("PHONE", "555-0101")], width=3)
    assert shown == [nb.spans_in_context(TEXT, [_span("PHONE", "555-0101")], width=3)]


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"page": 3}, "page 3"),
        ({"sheet": "Q1", "cell": "B7"}, "Q1!B7"),
        ({"row": 4, "column": "email"}, "row 4, email"),
        ({"part": "header", "name": "From"}, "header From"),
        ({"part": "body", "line": 2}, "body, line 2"),
        ({"part": "footer", "section": 0}, "footer, section 0"),
        ({"paragraph": 12}, "paragraph 12"),
        ({"table": 0, "row": 1, "col": 2}, "table 0 row 1 col 2"),
        ({"content_control": 0, "paragraph": 3}, "content control 0 › paragraph 3"),
        ({"content_control": 1, "table": 0, "row": 1, "col": 2}, "content control 1 › table 0 row 1 col 2"),
        ({"part": "header", "section": 0, "variant": "first"}, "header, section 0, variant first"),
        ({"line": 7}, "line 7"),
        ({"attachment": "x.pdf", "page": 1}, "x.pdf › page 1"),
        ({"attachment": "x.txt"}, "x.txt"),
        ([{"page": 1}, {"page": 2}], "page 1; page 2"),
        (None, ""),
        ([], ""),
    ],
)
def test_format_where(where, expected):
    assert nb.format_where(where) == expected
