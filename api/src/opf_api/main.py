from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

# Triton has no stable Apple Silicon support — set before any opf imports.
os.environ.setdefault("OPF_MOE_TRITON", "0")

from fastapi import FastAPI  # noqa: E402

from opf._api import OPF  # noqa: E402

from .routes import router  # noqa: E402


logger = logging.getLogger("opf_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = os.environ.get("OPF_DEVICE", "cpu")
    decode_mode = os.environ.get("OPF_DECODE_MODE", "viterbi")
    output_mode = os.environ.get("OPF_OUTPUT_MODE", "typed")
    logger.info("loading OPF: device=%s decode=%s output=%s", device, decode_mode, output_mode)
    opf = OPF(
        device=device,  # type: ignore[arg-type]
        decode_mode=decode_mode,  # type: ignore[arg-type]
        output_mode=output_mode,  # type: ignore[arg-type]
    )
    opf.get_runtime()  # warm load
    app.state.opf = opf
    app.state.lock = asyncio.Lock()
    logger.info("OPF ready")
    yield


app = FastAPI(title="OPF Privacy Filter API", version="0.1.0", lifespan=lifespan)
app.include_router(router)
