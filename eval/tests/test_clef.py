"""Clef-flash backend (plan 13, Part 6).

Fast tests cover `plan_load`, the loader import, the refusals that happen
before any download, and `ClefBackend.answer` batching against a fake model.

The slow tests build a *tiny random* Qwen3.5 backbone and joint schema head
in a temporary "release folder" and drive `ClefBackend` end to end through
the model repo's real `joint_schema_model.py`. They fetch only that file
and `config.json` (about 26 KB, pinned to one commit) and skip when the Hub
is unreachable. They are opt-in because they reach the network: set
`PII_BENCH_SLOW_TESTS=1` to run them (see `conftest.py`). The tokenizer is a
byte-level stand-in built locally. Set `PII_BENCH_CLEF_TOKENIZER=1` as well to
also run with the real 20 MB tokenizer. The 9.4B model itself is never
downloaded.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from opf_eval.io import meta_path, read_jsonl, write_jsonl
from opf_eval.review import DecisionBackend, clef
from opf_eval.review import classifier as rc

LABELS = ["EMAIL", "PERSON", "PHONE"]
# The model repo commit the slow tests were written against.
CLEF_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"


# ---------------------------------------------------------------- plan_load


@pytest.mark.parametrize(
    ("device", "vram", "bf16", "mode", "dtype", "quantize"),
    [
        ("cuda", 22.5, True, "bf16", "bfloat16", None),     # L4
        ("cuda", 22.0, True, "bf16", "bfloat16", None),     # L4 reporting the low end
        ("cuda", 21.0, True, "bf16", "bfloat16", None),     # the threshold itself
        ("cuda", 20.9, True, "int8", "bfloat16", "int8"),   # 8-bit weights, native bf16 compute
        ("cuda", 79.0, True, "bf16", "bfloat16", None),     # A100 80 GB
        ("cuda", 31.7, False, "fp16", "float16", None),     # V100 32 GB: no native bf16
        ("cuda", 15.0, False, "int8", "float16", "int8"),   # T4
        ("cuda", 12.0, True, "int8", "bfloat16", "int8"),
        ("cuda", 16.0, False, "int8", "float16", "int8"),
        ("cuda", 11.9, True, "too-small", "float16", None),
        ("cuda:1", 40.0, True, "bf16", "bfloat16", None),
        ("mps", 36.0, None, "bf16", "bfloat16", None),
        ("mps", 24.0, None, "too-small", "bfloat16", None),
        ("cpu", None, None, "cpu-too-slow", "float32", None),
    ],
)
def test_plan_load_rules(device, vram, bf16, mode, dtype, quantize):
    plan = clef.plan_load(device, vram, bf16=bf16)
    assert (plan["mode"], plan["dtype"], plan["quantize"]) == (mode, dtype, quantize)
    assert set(plan) == {"mode", "dtype", "quantize", "reason", "device", "vram_gb"}
    assert plan["reason"]


def test_plan_load_reads_the_machine(monkeypatch):
    monkeypatch.setattr(clef, "gpu_memory_gb", lambda index=0: 15.0)
    monkeypatch.setattr(clef, "_cuda_bf16", lambda index=0: False)
    assert clef.plan_load("cuda")["mode"] == "int8"
    monkeypatch.setattr(clef, "gpu_memory_gb", lambda index=0: None)
    assert clef.plan_load("cuda")["reason"] == "no CUDA device found"
    monkeypatch.setattr(clef, "_unified_memory_gb", lambda: 64.0)
    assert clef.plan_load("mps") == {
        "device": "mps", "vram_gb": 64.0, "mode": "bf16", "dtype": "bfloat16", "quantize": None,
        "reason": "64 GiB of unified memory holds the bf16 weights (unverified on MPS)",
    }
    monkeypatch.setattr(clef, "_unified_memory_gb", lambda: None)
    assert "unknown" in clef.plan_load("mps")["reason"]
    monkeypatch.setattr(clef, "_default_device", lambda: "cpu")
    assert clef.plan_load()["mode"] == "cpu-too-slow"


def _fake_cuda_torch(monkeypatch, cards):
    """Replace torch with a fake whose CUDA cards are `[(name, GiB, capability)]`.

    Like torch 2.3 and later, `is_bf16_supported()` counts emulation by default
    and so says True on every card."""
    import types

    def is_bf16_supported(including_emulation=True):
        return True if including_emulation else all(c[2][0] >= 8 for c in cards)

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=int(cards[i][1] * 1024**3)),
        get_device_name=lambda i: cards[i][0],
        get_device_capability=lambda i: cards[i][2],
        is_bf16_supported=is_bf16_supported,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_plan_load_reads_the_card_that_device_names(monkeypatch):
    _fake_cuda_torch(monkeypatch, [("Tesla T4", 14.7, (7, 5)), ("NVIDIA L4", 22.0, (8, 9))])
    assert clef.plan_load("cuda:0")["mode"] == "int8"
    plan = clef.plan_load("cuda:1")
    assert plan["mode"] == "bf16" and plan["vram_gb"] == pytest.approx(22.0)


def test_plan_load_ignores_emulated_bf16(monkeypatch):
    # A V100 32 GB has the memory but no native bf16. torch's default
    # is_bf16_supported() says True because it counts emulation.
    _fake_cuda_torch(monkeypatch, [("Tesla V100-SXM2-32GB", 31.7, (7, 0))])
    assert clef.plan_load("cuda")["mode"] == "fp16"


@pytest.mark.parametrize("gib", [22.0, 22.5, 21.4])
def test_plan_load_and_gpu_summary_agree_on_an_l4(monkeypatch, gib):
    from opf_eval import nb

    _fake_cuda_torch(monkeypatch, [("NVIDIA L4", gib, (8, 9))])
    monkeypatch.setattr(nb, "device", lambda prefer=None: "cuda")
    summary = nb.gpu_summary()
    plan = clef.plan_load("cuda")
    assert summary["bf16"] is True and plan["mode"] == "bf16"
    assert summary["vram_gb"] == round(plan["vram_gb"], 1)
    # Feeding the notebook's summary back in gives the same plan.
    assert clef.plan_load("cuda", summary["vram_gb"], bf16=summary["bf16"])["mode"] == "bf16"


def test_gpu_memory_gb_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert clef.gpu_memory_gb() is None


# ------------------------------------------------------- refusals and loader


def _no_download(*_a, **_k):
    raise AssertionError("must not download")


def test_backend_refuses_before_downloading(monkeypatch):
    monkeypatch.setattr(clef, "_snapshot", _no_download)
    with pytest.raises(RuntimeError, match="no GPU"):
        clef.ClefBackend(device="cpu")
    with pytest.raises(ValueError, match="unknown mode"):
        clef.ClefBackend(device="cpu", mode="fp8")
    with pytest.raises(ValueError, match="loader"):
        clef.ClefBackend(device="cpu", mode="fp32", loader="magic")


FAKE_LOADER = "VALUE = {value!r}\nQUESTION_TYPES = {{'noul': 0, 'choice': 1, 'score': 2}}\n"


def test_import_loader_keeps_copies_apart(tmp_path):
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        (tmp_path / name / clef.LOADER_FILE).write_text(FAKE_LOADER.format(value=name))
    one = clef.import_loader(tmp_path / "one")
    two = clef.import_loader(tmp_path / "two")
    assert (one.VALUE, two.VALUE) == ("one", "two")
    assert one.__name__ != two.__name__ and one.__name__ in sys.modules
    assert clef.import_loader(tmp_path / "one") is one  # imported once
    with pytest.raises(FileNotFoundError, match="Clef release"):
        clef.import_loader(tmp_path)


def test_mode_settings():
    assert clef._mode_settings("int8", True) == ("bfloat16", "int8")
    assert clef._mode_settings("int8", False) == ("float16", "int8")
    assert clef._mode_settings("fp16") == ("float16", None)


def test_int8_checks_bitsandbytes_before_downloading(monkeypatch):
    # A 16 GB card plans int8. Without bitsandbytes the backend must fail
    # before the 19 GB snapshot download, not after it.
    monkeypatch.setattr(clef, "gpu_memory_gb", lambda index=0: 15.6)
    monkeypatch.setattr(clef, "_cuda_bf16", lambda index=0: True)
    monkeypatch.setattr(clef, "_check_transformers", lambda: None)
    monkeypatch.setattr(clef, "_snapshot", _no_download)

    def missing():
        raise ImportError("8-bit loading needs bitsandbytes (Linux + CUDA): " + clef.INSTALL_HINT)

    monkeypatch.setattr(clef, "_require_bitsandbytes", missing)
    with pytest.raises(ImportError, match="bitsandbytes"):
        clef.ClefBackend(device="cuda")
    with pytest.raises(ImportError, match="bitsandbytes"):
        clef.ClefBackend(device="cuda", mode="int8")


def test_int8_without_bitsandbytes_names_the_extra(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("bitsandbytes is installed")
    (tmp_path / clef.LOADER_FILE).write_text(FAKE_LOADER.format(value="x"))
    monkeypatch.setattr(clef, "_check_transformers", lambda: None)
    with pytest.raises(ImportError, match=r"opf-eval\[clef\]"):
        clef.ClefBackend(tmp_path, device="cpu", mode="int8")


# --------------------------------------------- answer() with a fake model


class _FakeModel:
    """Stands in for ClefModel: logits favour option 0 by the record's length."""

    def __init__(self):
        import torch

        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.batches: list[list[int]] = []

    def parameters(self):
        yield self.weight

    def __call__(self, batch):
        import torch

        self.batches.append([len(r.input_ids) for r in batch["records"]])
        out = []
        for r in batch["records"]:
            out.append([torch.tensor([float(len(r.input_ids))] + [0.0] * (len(q.option_ids) - 1)) for q in r.questions])
        return out


def _fake_module():
    def encode_record(tokenizer, record, max_length=16384, processor=None):
        n = len(str(record["state"]))
        questions = []
        for qid, q in record["questions"].items():
            options = ["true", "false"] if q["type"] == "noul" else sorted(q["criteria"])
            questions.append(SimpleNamespace(question_id=qid, option_ids=tuple(options)))
        return SimpleNamespace(input_ids=tuple(range(n)), questions=tuple(questions))

    def collate_records(records, pad_id, device):
        return {"records": records, "pad": pad_id}

    def systemone_answer(question, probabilities):
        if question["type"] == "noul":
            return {"type": "noul", "noul": round(probabilities["true"], 4)}
        best = max(probabilities, key=probabilities.__getitem__)
        return {"type": "choice", "choice": best, "confidence": round(probabilities[best], 4)}

    return SimpleNamespace(
        QUESTION_TYPES={"noul": 0, "choice": 1, "score": 2},
        encode_record=encode_record,
        collate_records=collate_records,
        systemone_answer=systemone_answer,
    )


def test_answer_sorts_batches_and_restores_order():
    model = _FakeModel()
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=7)
    backend = clef.ClefBackend.from_components(model, tokenizer, _fake_module(), batch_size=2)
    assert isinstance(backend, DecisionBackend)
    assert backend._pad_id() == 7
    texts = ["a" * 30, "b" * 5, "c" * 50, "d" * 10]
    reqs = [rc.segment_requests(t, [], LABELS)[0][2] for t in texts]
    answers = backend.answer(reqs)
    # Shortest first, two per forward pass.
    lengths = [len(str(r["state"])) for r in reqs]
    assert model.batches == [sorted(lengths)[:2], sorted(lengths)[2:]]
    # Answers come back in request order, each with usage and both questions.
    assert [a["usage"]["input_tokens"] for a in answers] == lengths
    assert all(set(a["answers"]) == {"residual", "residual_type"} for a in answers)
    assert answers[0]["answers"]["residual"]["noul"] == 1.0  # logit 30 vs 0
    assert answers[0]["model"] == rc.DEFAULT_MODEL
    with pytest.raises(ValueError, match="criteria"):
        backend.answer([{"model": "m", "state": "x", "questions": {"q": {"type": "choice"}}}])
    with pytest.raises(ValueError, match="state"):
        backend.answer([{"model": "m", "questions": {"q": {"type": "noul"}}}])
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.answer(reqs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_raise_instead_of_becoming_probabilities(bad, tmp_path):
    import torch

    class Overflow(_FakeModel):
        def __call__(self, batch):
            return [[torch.tensor([bad, 0.0], dtype=torch.float16) if q.question_id == "residual"
                     else torch.zeros(len(q.option_ids)) for q in r.questions] for r in batch["records"]]

    backend = clef.ClefBackend.from_components(Overflow(), SimpleNamespace(pad_token_id=0), _fake_module())
    req = rc.segment_requests("Call Ann today.", [], LABELS)[0][2]
    with pytest.raises(FloatingPointError, match="non-finite"):
        backend.answer([req])


def test_close_frees_the_weights_even_when_the_model_is_still_referenced(monkeypatch):
    import torch

    model = torch.nn.Linear(4, 4)
    key = ("/clef", "cpu", "fp32")
    monkeypatch.setitem(clef._CACHE, key, (model, None, _fake_module()))
    monkeypatch.setitem(clef._USERS, key, 2)

    def backend():
        b = clef.ClefBackend.from_components(model, None, _fake_module())
        b._key = key
        return b

    first, second = backend(), backend()
    held = model  # like a traceback frame that still points at the model
    first.close()
    assert held.weight is not None  # the second backend still uses it
    assert key in clef.loaded()
    second.close()
    assert held.weight is None and held.bias is None
    assert key not in clef.loaded() and key not in clef._USERS
    second.close()  # closing twice is harmless


# ------------------------------------------------- tiny real model (slow)


def _hub_file(name: str) -> Path:
    hub = pytest.importorskip("huggingface_hub")
    try:
        return Path(hub.hf_hub_download(clef.CLEF_FLASH, name, revision=CLEF_REVISION))
    except Exception as exc:  # noqa: BLE001 — offline or rate limited
        pytest.skip(f"cannot fetch {name} from {clef.CLEF_FLASH}: {exc}")


def _stub_tokenizer(folder: Path):
    """A byte-level BPE with no merges plus the Qwen chat and vision special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
        "<|image_pad|>", "<|video_pad|>", "<think>", "</think>",
    ])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<|endoftext|>", eos_token="<|im_end|>")
    tok.save_pretrained(folder)
    return tok


def _real_tokenizer(folder: Path):
    from transformers import AutoTokenizer

    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(_hub_file(name), folder / name)
    return AutoTokenizer.from_pretrained(folder)


def build_tiny_release(folder: Path, *, real_tokenizer: bool = False) -> Path:
    """A Clef release folder with a random 2-layer Qwen3.5 backbone and a 1-layer head."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        pytest.skip(f"transformers {transformers.__version__} has no Qwen3.5")
    from safetensors.torch import save_file

    folder.mkdir(parents=True)
    shutil.copy(_hub_file(clef.LOADER_FILE), folder / clef.LOADER_FILE)
    cfg = json.loads(_hub_file("config.json").read_text())
    tok = _real_tokenizer(folder) if real_tokenizer else _stub_tokenizer(folder)

    t = cfg["text_config"]
    t.update(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], full_attention_interval=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        vocab_size=max(512, len(tok)), eos_token_id=tok.eos_token_id,
    )
    t["rope_parameters"]["mrope_section"] = [2, 1, 1]  # sums to head_dim * partial_rotary_factor / 2
    cfg["vision_config"].update(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                                out_hidden_size=32, num_position_embeddings=16)
    config = transformers.Qwen3_5Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "transformers_version")})
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(config).save_pretrained(folder)

    module = clef.import_loader(folder)
    head_cfg = {"hidden_size": 32, "width": 16, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 32}
    head = module.JointSchemaHead(**head_cfg)
    with torch.no_grad():  # the release head starts its gates at 0; make the logits depend on the input
        head.prior_logit_scale.fill_(1.0)
        head.joint_logit_scale.fill_(1.0)
        head.residual_gate.fill_(1.0)
    (folder / "joint_head_config.json").write_text(json.dumps(head_cfg))
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(folder / "joint_head.safetensors"))
    return folder


@pytest.fixture(scope="module")
def tiny_release(tmp_path_factory):
    return build_tiny_release(tmp_path_factory.mktemp("clef") / "tiny-clef")


TEXT = "Dear Jane Doe,\nplease call 555-1234 or mail jane@x.com.\nThanks."


def _requests() -> list[dict]:
    span = {"start": 5, "end": 13, "label": "PERSON"}
    reqs = [rc.span_request(TEXT, span, LABELS)]
    reqs += [r for _, _, r in rc.segment_requests(TEXT, [span], LABELS)]
    reqs.append({"model": "clef-flash", "state": {"x": 1},
                 "questions": {"urgency": {"type": "score", "criteria": ["low", "high"]}}})
    return reqs


def _check_answers(reqs: list[dict], answers: list[dict]) -> None:
    assert len(answers) == len(reqs)
    for req, res in zip(reqs, answers):
        assert set(res["answers"]) == set(req["questions"])
        for qid, a in res["answers"].items():
            q = req["questions"][qid]
            assert a["type"] == q["type"]
            if q["type"] == "noul":
                assert 0.0 <= a["noul"] <= 1.0
            elif q["type"] == "choice":
                assert a["choice"] in q["criteria"]
                assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        assert res["usage"]["input_tokens"] > 0


@pytest.mark.slow
def test_tiny_clef_batched_answers_match_single_requests(tiny_release):
    backend = clef.ClefBackend(tiny_release, device="cpu", mode="fp32", loader="text", batch_size=3)
    try:
        reqs = _requests()
        answers = backend.answer(reqs)
        _check_answers(reqs, answers)
        # The repo's own one-request path must give the same probabilities, so
        # padding and sorting in our batches change nothing.
        proc = SimpleNamespace(tokenizer=backend.tokenizer)
        for req, ours in zip(reqs, answers):
            ref = backend.module.systemone(backend.model, proc, req, max_length=backend.max_length)
            for qid, a in ref["answers"].items():
                got = ours["answers"][qid]
                for key in ("noul", "confidence", "score"):
                    if key in a:
                        assert got[key] == pytest.approx(a[key], abs=2e-4), (qid, key)
                if "probabilities" in a:
                    for opt, p in a["probabilities"].items():
                        assert got["probabilities"][opt] == pytest.approx(p, abs=2e-4)
        # Loaded once per process: a second backend reuses the model.
        again = clef.ClefBackend(tiny_release, device="cpu", mode="fp32", loader="text")
        assert again.model is backend.model
        assert clef.loaded()
        # Closing one backend leaves the model working for the other.
        again.close()
        assert backend.answer(reqs[:1])[0]["answers"]
    finally:
        backend.close()
    assert not any(k[0] == str(tiny_release.resolve()) for k in clef.loaded())


@pytest.mark.slow
def test_tiny_clef_through_the_release_loader(tiny_release, monkeypatch):
    """`load_release_model` builds an AutoProcessor, which needs torchvision for
    video. Standing a tokenizer in for it runs the repo's loader end to end."""
    import transformers

    monkeypatch.setattr(transformers, "AutoProcessor", transformers.AutoTokenizer)
    text_backend = clef.ClefBackend(tiny_release, device="cpu", mode="fp32", loader="text")
    expected = text_backend.answer(_requests())
    text_backend.close()
    backend = clef.ClefBackend(tiny_release, device="cpu", mode="fp32", loader="release")
    try:
        assert backend.answer(_requests()) == expected
    finally:
        backend.close()


@pytest.mark.slow
def test_tiny_clef_review_run(tiny_release, tmp_path):
    fixtures = tmp_path / "fx.jsonl"
    write_jsonl(fixtures, [{"id": "r1", "text": TEXT, "language": None, "gold_spans": []}])
    meta_path(fixtures).write_text(json.dumps({"gold": "none", "labels": LABELS}))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"fixtures": str(fixtures), "labels": LABELS, "detectors": ["a"]}))
    start = TEXT.index("jane@")
    write_jsonl(run_dir / "raw_a.jsonl", [{"id": "r1", "detector": "a", "error": None, "latency_ms": 1.0, "spans": [
        {"label": "EMAIL", "fine_label": "EMAIL", "raw_label": "email", "start": start, "end": start + 10, "text": "jane@x.com"},
    ]}])
    backend = clef.ClefBackend(tiny_release, device="cpu", mode="fp32", loader="text")
    try:
        path = rc.review_run(run_dir, backend, progress=False, batch_size=2)
    finally:
        backend.close()
    rows = read_jsonl(path)
    assert path.name == "review_clef_flash.jsonl"
    assert [r["kind"] for r in rows] == ["span", "segment", "segment", "segment"]
    assert all(0.0 <= r.get("p_pii", r.get("p_residual")) <= 1.0 for r in rows)
    (row,) = rc.summarize(run_dir, path)
    assert row["n_spans"] == 1 and row["n_segments"] == 3


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("PII_BENCH_CLEF_TOKENIZER") != "1", reason="set PII_BENCH_CLEF_TOKENIZER=1 to fetch the 20 MB tokenizer")
def test_tiny_clef_with_the_real_tokenizer(tmp_path):
    folder = build_tiny_release(tmp_path / "tiny-real", real_tokenizer=True)
    backend = clef.ClefBackend(folder, device="cpu", mode="fp32", loader="text")
    try:
        reqs = _requests()
        _check_answers(reqs, backend.answer(reqs))
        assert backend._pad_id() == backend.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    finally:
        backend.close()


def test_default_device_follows_pii_bench_device(monkeypatch):
    # Same rule as nb.device(), so nb.gpu_summary and plan_load look at the same card.
    from opf_eval import nb

    monkeypatch.setenv("PII_BENCH_DEVICE", "cuda:1")
    assert clef._default_device() == nb.device() == "cuda:1"
    _fake_cuda_torch(monkeypatch, [("Tesla T4", 14.7, (7, 5)), ("NVIDIA L4", 22.0, (8, 9))])
    assert clef.plan_load()["mode"] == "bf16"
    assert nb.gpu_summary()["name"] == "NVIDIA L4"
