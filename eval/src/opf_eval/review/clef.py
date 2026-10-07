"""Cloudflare Clef-flash as a local `DecisionBackend` (plan 13, Part 6).

[Clef-flash](https://huggingface.co/Cloudflare/clef-flash) is a 9.4B Qwen3.5
backbone with a small joint schema head. It reads a SystemOne request and
returns a probability for every option of every question in one forward
pass, with no text generation. The weights are Apache-2.0 and run locally,
so no text leaves the machine.

    from opf_eval.review import classifier, clef

    print(clef.plan_load())        # {"mode": "bf16", "dtype": "bfloat16", ...} on an L4
    backend = clef.ClefBackend()   # downloads ~19 GB once, then loads once per process
    path = classifier.review_run(run_dir, backend)
    backend.close()                # frees the GPU memory

Memory: the bf16 weights take about 18.8 GB (17.5 GiB). The thresholds in
`plan_load` are in GiB because that is what torch reports, and they use the
same helpers as `opf_eval.nb.gpu_summary` (`opf_eval.hardware`), so the
notebook's GPU summary and the load plan always agree about a card.

| runtime                          | plan_load mode | notes                                |
| -------------------------------- | -------------- | ------------------------------------ |
| CUDA >= 21 GiB with native bf16  | bf16           | primary target                       |
| (L4, A100, H100)                 |                |                                      |
| CUDA >= 21 GiB without native    | fp16           | float16 weights fit; float16         |
| bf16 (V100)                      |                | activations of a bf16 model can      |
|                                  |                | overflow and such requests become    |
|                                  |                | error rows                           |
| CUDA 12 to 21 GiB (T4 reports    | int8           | 8-bit weights via bitsandbytes       |
| ~15 GiB)                         |                | (about 10.7 GiB) with bfloat16       |
|                                  |                | compute where the card has it and    |
|                                  |                | float16 compute otherwise;           |
|                                  |                | probabilities may shift slightly     |
|                                  |                | against bf16                         |
| CUDA < 12 GiB                    | too-small      |                                      |
| Apple Silicon >= 32 GiB unified  | bf16           | unverified on MPS                    |
| memory                           |                |                                      |
| Apple Silicon < 32 GiB unified   | too-small      |                                      |
| memory                           |                |                                      |
| CPU                              | cpu-too-slow   |                                      |

The model repo ships its own loader, `joint_schema_model.py`. `ClefBackend`
imports that file from the downloaded snapshot and uses its
`encode_record`, `collate_records`, `ClefModel` and `systemone_answer`, so
prompts are encoded exactly as the model was trained. Its
`load_release_model` builds an `AutoProcessor` for image and video input,
which needs torchvision. Reviews only send text, so when torchvision is
missing the backend loads the same backbone and head with a plain tokenizer
instead (`loader="text"`).

torch, transformers and huggingface_hub are imported only when a backend is
built: `pip install 'opf-eval[clef]'`.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from .. import hardware
from .classifier import (  # noqa: F401 — re-exported
    StubBackend,
    choice_answer,
    noul_answer,
)

CLEF_FLASH = "Cloudflare/clef-flash"
LOADER_FILE = "joint_schema_model.py"
INSTALL_HINT = "pip install 'opf-eval[clef]'"

# Thresholds in GiB (what torch reports). An L4 reports about 22.0 to 22.5 GiB
# and a T4 about 14.7 to 15 GiB. The bf16 threshold sits below the smallest L4
# reading so that every L4 loads in bf16, and above the 17.5 GiB of weights.
BF16_MIN_GB = 21.0
INT8_MIN_GB = 12.0
MPS_MIN_GB = 32.0
MODES = ("bf16", "fp16", "int8", "fp32")

# One loaded model per (path, device, mode) per process: the weights are 19 GB
# and the notebook may build several backends. _USERS counts the open
# backends of each entry so the last close() can free the weights.
_CACHE: dict[tuple[str, str, str], tuple[Any, Any, ModuleType]] = {}
_USERS: dict[tuple[str, str, str], int] = {}
_CACHE_LOCK = threading.Lock()


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover — torch is a core dependency
        raise ImportError(f"Clef-flash needs torch: {INSTALL_HINT}") from exc
    return torch


def gpu_memory_gb(index: int = 0) -> float | None:
    """Total memory of CUDA device `index` in GiB, or None without CUDA (or torch)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return hardware.cuda_memory_gib(index)


def _default_device() -> str:
    """$PII_BENCH_DEVICE when set (as in `opf_eval.nb.device`), else cuda > mps > cpu."""
    env = os.environ.get("PII_BENCH_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _cuda_bf16(index: int = 0) -> bool:
    """Native (not emulated) bfloat16 on CUDA device `index`, as in `nb.gpu_summary`."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and hardware.cuda_native_bf16(index))


def _unified_memory_gb() -> float | None:
    return hardware.unified_memory_gib()


def plan_load(
    device: str | None = None, vram_gb: float | None = None, *, bf16: bool | None = None
) -> dict:
    """How to load Clef-flash on this machine, without loading anything.

    Returns `{"mode", "dtype", "quantize", "reason", "device", "vram_gb"}`:

    1. `mode` is "bf16" (load as is), "fp16" (float16 weights), "int8"
       (8-bit weights through bitsandbytes, computing in `dtype`),
       "too-small" or "cpu-too-slow".
    2. `device` defaults to cuda, then mps, then cpu. `vram_gb` (GiB)
       defaults to the memory of the CUDA card that `device` names ("cuda:1"
       reads card 1) or on mps to the unified memory.
    3. `bf16` overrides the check for native bfloat16 support (CUDA only).

    Rules: CUDA with >= 21 GiB and native bf16 -> bf16. CUDA with >= 21 GiB
    and no native bf16 -> fp16 because the float16 weights fit and 8-bit
    weights would still compute in float16. CUDA with 12-21 GiB -> int8
    with bfloat16 compute on a card with native bf16 and float16 compute on
    one without. Less -> too-small. MPS with >= 32 GiB unified memory ->
    bf16, else too-small. CPU -> cpu-too-slow. Float16 activations of a
    model trained in bf16 can overflow. `ClefBackend.answer` raises on a
    non-finite output, so such a request becomes an error row and never a
    silent probability.
    """
    dev, _, idx = (device or _default_device()).partition(":")
    index = int(idx) if idx.isdigit() else 0
    if dev == "cuda":
        vram = vram_gb if vram_gb is not None else gpu_memory_gb(index)
        native = _cuda_bf16(index) if bf16 is None else bf16
        base = {"device": "cuda", "vram_gb": vram}
        if vram is None:
            return {**base, "mode": "too-small", "dtype": "float16", "quantize": None,
                    "reason": "no CUDA device found"}
        if vram >= BF16_MIN_GB and native:
            return {**base, "mode": "bf16", "dtype": "bfloat16", "quantize": None,
                    "reason": f"{vram:.1f} GiB with native bf16 holds the 17.5 GiB of bf16 "
                              "weights"}
        if vram >= BF16_MIN_GB:
            return {**base, "mode": "fp16", "dtype": "float16", "quantize": None,
                    "reason": f"{vram:.1f} GiB without native bf16 holds the 17.5 GiB of "
                              "float16 weights. Float16 activations of a bf16 model can overflow; "
                              "such requests become error rows."}
        if vram >= INT8_MIN_GB:
            compute = "bfloat16" if native else "float16"
            return {**base, "mode": "int8", "dtype": compute, "quantize": "int8",
                    "reason": f"{vram:.1f} GiB is too small for the 17.5 GiB of bf16 weights; "
                              "8-bit weights need about 10.7 GiB "
                              f"(bitsandbytes, Linux + CUDA only) and compute in {compute}. "
                              "Probabilities may shift slightly against a bf16 run."}
        return {**base, "mode": "too-small", "dtype": "float16", "quantize": None,
                "reason": f"{vram:.1f} GiB is below the {INT8_MIN_GB:.0f} GiB that 8-bit "
                          "weights need; use an L4 or larger runtime"}
    if dev == "mps":
        mem = vram_gb if vram_gb is not None else _unified_memory_gb()
        base = {"device": "mps", "vram_gb": mem}
        if mem is not None and mem >= MPS_MIN_GB:
            return {**base, "mode": "bf16", "dtype": "bfloat16", "quantize": None,
                    "reason": f"{mem:.0f} GiB of unified memory holds the bf16 weights "
                              "(unverified on MPS)"}
        shown = "unknown" if mem is None else f"{mem:.0f} GiB"
        return {**base, "mode": "too-small", "dtype": "bfloat16", "quantize": None,
                "reason": f"{shown} of unified memory; Clef-flash needs at least "
                          f"{MPS_MIN_GB:.0f} GiB on Apple Silicon"}
    return {"device": dev, "vram_gb": None, "mode": "cpu-too-slow", "dtype": "float32",
            "quantize": None,
            "reason": "no GPU: a 9.4B model on CPU takes minutes per request; use a GPU "
                      "runtime (L4)"}


def _mode_settings(mode: str, native_bf16: bool = False) -> tuple[str, str | None]:
    """(dtype name, quantize) for an explicit mode. 8-bit weights compute in
    bfloat16 on a card with native bf16 and in float16 otherwise."""
    if mode == "bf16":
        return "bfloat16", None
    if mode == "fp16":
        return "float16", None
    if mode == "int8":
        return ("bfloat16" if native_bf16 else "float16"), "int8"
    if mode == "fp32":
        return "float32", None
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def _require_bitsandbytes() -> None:
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"8-bit loading needs bitsandbytes (Linux + CUDA): {INSTALL_HINT}"
        ) from exc


def import_loader(path: str | Path) -> ModuleType:
    """Import `joint_schema_model.py` from a model folder as its own module.

    Each folder gets a module name derived from its path, so two snapshots
    (or a test copy) never shadow each other the way a bare
    `sys.path.insert` + `import joint_schema_model` would.
    """
    file = Path(path) / LOADER_FILE
    if not file.exists():
        raise FileNotFoundError(f"{file} not found; is {path} a Clef release folder?")
    name = "_clef_joint_schema_" + hashlib.sha1(str(file.resolve()).encode()).hexdigest()[:10]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses look their module up in sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _snapshot(model_id: str | Path, revision: str | None) -> Path:
    path = Path(model_id)
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(f"downloading {model_id} needs huggingface_hub: {INSTALL_HINT}") from exc
    return Path(snapshot_download(str(model_id), revision=revision))


def _has_torchvision() -> bool:
    return importlib.util.find_spec("torchvision") is not None


def _load_text_only(
    module: ModuleType, path: Path, device: str, dtype, **kwargs
) -> tuple[Any, Any]:
    """`load_release_model` without the image/video processor.

    Same backbone, same head and same tokenizer, but the tokenizer comes
    from `AutoTokenizer`, which does not need torchvision. Records with
    images or videos cannot be encoded this way; reviews never send any.
    """
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=dtype, device_map={"": str(device)}, **kwargs
    )
    backbone.config.use_cache = False
    head = module.JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return module.ClefModel(backbone, head).eval(), tokenizer


def _check_transformers() -> None:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(f"Clef-flash needs transformers: {INSTALL_HINT}") from exc
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        raise ImportError(
            f"transformers {transformers.__version__} has no Qwen3.5 support; "
            f"Clef-flash needs transformers >= 5.10.2: {INSTALL_HINT}"
        )


class ClefBackend:
    """A local `DecisionBackend` running Clef-flash.

    model_id: a Hugging Face repo id or a local folder holding a Clef release
        (weights, `joint_head.safetensors`, `joint_head_config.json`,
        tokenizer files and `joint_schema_model.py`).
    device: "cuda", "mps" or "cpu" (default: the best available).
    mode: "bf16", "fp16", "int8" or "fp32". Default: what `plan_load` picks; it
        refuses "too-small" and "cpu-too-slow" with the reason, so pass a
        mode explicitly to override (fp32 on CPU is meant for tests).
    max_length: token limit per request; longer states are truncated by the
        model's own encoder.
    revision: a commit or tag of the model repo, to pin the weights.
    batch_size: requests per forward pass inside `answer`.
    loader: "release" uses the repo's `load_release_model`, "text" loads with
        a plain tokenizer (no torchvision needed), "auto" picks "release"
        when torchvision is installed.
    """

    name = "clef_flash"
    remote = False

    def __init__(
        self,
        model_id: str | Path = CLEF_FLASH,
        *,
        device: str | None = None,
        mode: str | None = None,
        max_length: int = 4096,
        revision: str | None = None,
        batch_size: int = 8,
        loader: str = "auto",
    ):
        if loader not in ("auto", "release", "text"):
            raise ValueError("loader must be 'auto', 'release' or 'text'")
        self.model_id = str(model_id)
        self.model_name = Path(self.model_id).name.lower() or "clef"
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.revision = revision
        self.device = device or _default_device()
        self._key = None
        self.model = self.processor = self.module = None
        if mode is None:
            plan = plan_load(self.device)
            if plan["mode"] not in MODES:
                raise RuntimeError(f"Clef-flash will not load on {self.device}: {plan['reason']}")
            mode = plan["mode"]
            dtype_name, quantize = plan["dtype"], plan["quantize"]
        else:
            dev, _, idx = self.device.partition(":")
            native = dev == "cuda" and _cuda_bf16(int(idx) if idx.isdigit() else 0)
            dtype_name, quantize = _mode_settings(mode, native)
        self.mode = mode
        self.dtype = dtype_name
        _check_transformers()
        if quantize == "int8":
            _require_bitsandbytes()  # before the 19 GB download, not after it
        torch = _torch()

        self.path = _snapshot(model_id, revision)
        key = (str(self.path.resolve()), self.device, mode)
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
            if cached is None:
                module = import_loader(self.path)
                kwargs: dict[str, Any] = {}
                if quantize == "int8":
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                dtype = getattr(torch, dtype_name)
                use_release = loader == "release" or (loader == "auto" and _has_torchvision())
                if use_release:
                    model, processor = module.load_release_model(
                        self.path, device=self.device, dtype=dtype, **kwargs
                    )
                else:
                    model, processor = _load_text_only(
                        module, self.path, self.device, dtype, **kwargs
                    )
                cached = _CACHE[key] = (model, processor, module)
            _USERS[key] = _USERS.get(key, 0) + 1
        self._key = key
        self.model, self.processor, self.module = cached

    @classmethod
    def from_components(
        cls,
        model: Any,
        processor: Any,
        module: ModuleType,
        *,
        name: str = "clef_flash",
        model_name: str = "clef-flash",
        max_length: int = 4096,
        batch_size: int = 8,
    ) -> ClefBackend:
        """A backend around an already loaded model, processor (or tokenizer) and loader module.

        For callers that load the model themselves, for example with a
        patched loader or a quantization config this class does not offer.
        `close()` on such a backend drops the references but leaves the
        module cache alone.
        """
        self = cls.__new__(cls)
        self.model_id = model_name
        self.model_name = model_name
        self.name = name
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.revision = None
        self.device = str(next(model.parameters()).device)
        self.mode = "custom"
        self.dtype = None
        self.path = None
        self._key = None
        self.model, self.processor, self.module = model, processor, module
        return self

    def __repr__(self) -> str:
        return f"ClefBackend({self.model_id!r}, device={self.device!r}, mode={self.mode!r})"

    @property
    def tokenizer(self) -> Any:
        return getattr(self.processor, "tokenizer", self.processor)

    def _pad_id(self) -> int:
        tok = self.tokenizer
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(tok, attr, None)
            if value is not None:
                return int(value)
        return 0

    def _validate(self, request: dict) -> None:
        questions = request.get("questions")
        if "state" not in request:
            raise ValueError("request has no state")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("request needs at least one question")
        for qid, q in questions.items():
            if q.get("type") not in self.module.QUESTION_TYPES:
                raise ValueError(f"{qid}: type must be noul, choice or score")
            if q["type"] != "noul" and not q.get("criteria"):
                raise ValueError(f"{qid}: criteria must not be empty")

    def answer(self, requests: list[dict]) -> list[dict]:
        """SystemOne responses for `requests`, in order.

        Requests are encoded with the repo's `encode_record`, sorted by length
        so each batch pads little, run `batch_size` at a time through one
        forward pass each, and turned into answers with a softmax per question
        and the repo's `systemone_answer`.
        """
        if self.model is None:
            raise RuntimeError("this ClefBackend was closed")
        torch = _torch()
        for req in requests:
            self._validate(req)
        processor = self.processor if self.tokenizer is not self.processor else None
        encoded = [
            self.module.encode_record(
                self.tokenizer, req, max_length=self.max_length, processor=processor
            )
            for req in requests
        ]
        order = sorted(range(len(requests)), key=lambda i: len(encoded[i].input_ids))
        device = next(self.model.parameters()).device
        out: list[dict | None] = [None] * len(requests)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            batch = self.module.collate_records([encoded[i] for i in idx], self._pad_id(), device)
            with torch.inference_mode():
                logits = self.model(batch)
            for i, record_logits in zip(idx, logits, strict=False):
                req, enc = requests[i], encoded[i]
                answers = {}
                for question, q_logits in zip(enc.questions, record_logits, strict=False):
                    q_logits = q_logits.float()
                    if not bool(torch.isfinite(q_logits).all()):
                        # A float16 overflow gives inf or NaN, and a NaN
                        # probability would read as "no PII" downstream.
                        raise FloatingPointError(
                            f"non-finite logits for question {question.question_id!r} "
                            f"(mode {self.mode}); the model overflowed on this request"
                        )
                    probs = q_logits.softmax(-1).tolist()
                    answers[question.question_id] = self.module.systemone_answer(
                        req["questions"][question.question_id],
                        dict(zip(question.option_ids, probs, strict=False)),
                    )
                out[i] = {
                    "model": req.get("model", self.model_name),
                    "answers": answers,
                    "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0},
                }
        return out  # type: ignore[return-value]

    def close(self) -> None:
        """Drop the model and free its GPU memory (it reloads on the next ClefBackend).

        The last open backend of a cached model also empties the model's
        parameters and buffers. That frees the weights even when something
        else still holds the model object, for example the traceback of an
        interrupted run that IPython keeps in `sys.last_traceback`. A model
        passed to `from_components` belongs to the caller and is left as it is.
        """
        model, key = self.model, self._key
        self.model = None
        self.processor = None
        self._key = None
        if key is not None:
            with _CACHE_LOCK:
                left = _USERS.get(key, 1) - 1
                if left > 0:
                    _USERS[key] = left
                    model = None  # another backend still uses it
                else:
                    _USERS.pop(key, None)
                    _CACHE.pop(key, None)
            if model is not None:
                _release(model)
        free_memory()


def _release(model: Any) -> None:
    """Drop every parameter and buffer of `model` so their memory is freed
    even while other references to the model object remain."""
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return
    for module in modules():
        for store in ("_parameters", "_buffers"):
            tensors = getattr(module, store, None)
            if isinstance(tensors, dict):
                for name in list(tensors):
                    tensors[name] = None


def free_memory() -> None:
    """Collect garbage and return cached GPU memory to the driver."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        empty: Callable[[], None] | None = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if callable(empty):
            empty()


def loaded() -> list[tuple[str, str, str]]:
    """(path, device, mode) of every Clef model held in this process."""
    with _CACHE_LOCK:
        return list(_CACHE)
