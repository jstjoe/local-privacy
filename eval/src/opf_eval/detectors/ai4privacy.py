"""ai4privacy multilingual ModernBERT-based PII anonymiser.

Model: `ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii`
(despite the `llama-` prefix in the name, the actual base is
`answerdotai/ModernBERT-base` per the model card).

Trained on `ai4privacy/open-pii-masking-500k-ai4privacy` — same OpenPII
vocabulary used by our `pii_masking_400k` / `openpii_nano` / `openpii_mini`
datasets, so labels map directly through the `openpii` vocabulary.

8 languages: fr, en, de, te, hi, it, es, nl. MIT license. ~150M params.

`aggregation_strategy="simple"` is empirically best on this model: ~3× more
entities with cleaner boundaries than `first`.
"""

from __future__ import annotations

from .hf_token import AggregationStrategy, HFTokenClassifierDetector

DEFAULT_MODEL = "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"


class Ai4PrivacyDetector(HFTokenClassifierDetector):
    name = "ai4privacy_modernbert"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        aggregation_strategy: AggregationStrategy = "simple",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            model_name=model_name,
            vocab="openpii",
            aggregation_strategy=aggregation_strategy,
            device=device,
        )
