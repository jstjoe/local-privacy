FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPF_MOE_TRITON=0 \
    OPF_DEVICE=cpu \
    OPF_DECODE_MODE=viterbi \
    OPF_OUTPUT_MODE=typed

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

WORKDIR /app

# Copy OPF source and api/ source. Build context must be the repo root
# (build with: `docker build -f api/Dockerfile -t opf-api .`).
COPY privacy-filter ./privacy-filter
COPY api ./api

RUN uv pip install --system -e ./privacy-filter -e ./api

# Bake the checkpoint into the image so first request doesn't pay the 2.8 GB
# download cost. If you'd rather mount it at runtime, drop this line and
# set OPF_CHECKPOINT in the container env.
RUN python -c "from opf._common.checkpoint_download import ensure_default_checkpoint; ensure_default_checkpoint()"

EXPOSE 8080
CMD ["uvicorn", "opf_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
