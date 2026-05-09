FROM python:3.12-slim

ARG INCLUDE_OPF=1
ARG INCLUDE_GLINER=0
ARG INCLUDE_PRESIDIO=0
ARG INCLUDE_PRESIDIO_MULTILANG=0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPF_MOE_TRITON=0 \
    OPF_DEVICE=cpu \
    OPF_DECODE_MODE=viterbi \
    DEFAULT_DETECTOR=opf \
    EAGER_LOAD=opf

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

WORKDIR /app

# Build context = repo root: `docker build -f api/Dockerfile -t privacy-api .`
# Copy the workspace root pyproject so uv resolves `workspace = true` sources
# in eval/ and api/ instead of falling back to PyPI for `opf`/`opf-eval`.
COPY pyproject.toml ./
COPY privacy-filter ./privacy-filter
COPY eval ./eval
COPY api ./api

# Always install the OPF source + eval (runs detectors) + api package.
# Optional extras pull in the per-detector dependencies.
RUN uv pip install --system -e ./privacy-filter -e ./eval -e ./api

RUN if [ "$INCLUDE_GLINER" = "1" ]; then \
        uv pip install --system gliner>=0.2.13; \
    fi

RUN if [ "$INCLUDE_PRESIDIO" = "1" ] || [ "$INCLUDE_PRESIDIO_MULTILANG" = "1" ]; then \
        uv pip install --system presidio-analyzer>=2.2 && \
        python -m spacy download en_core_web_lg; \
    fi

RUN if [ "$INCLUDE_PRESIDIO_MULTILANG" = "1" ]; then \
        python -m spacy download nl_core_news_lg && \
        python -m spacy download fr_core_news_lg && \
        python -m spacy download de_core_news_lg && \
        python -m spacy download it_core_news_lg && \
        python -m spacy download es_core_news_lg; \
    fi

# Bake OPF checkpoint into the image so first /v1/redact doesn't pay the
# 2.8 GB download cost. Skip with --build-arg INCLUDE_OPF=0.
RUN if [ "$INCLUDE_OPF" = "1" ]; then \
        python -c "from opf._common.checkpoint_download import ensure_default_checkpoint; ensure_default_checkpoint()"; \
    fi

# Bake GLiNER checkpoint into the image (~500 MB).
RUN if [ "$INCLUDE_GLINER" = "1" ]; then \
        python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi_pii-v1')"; \
    fi

# Cloud Run injects $PORT (default 8080); fall back to 8080 for local docker run.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn opf_api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
