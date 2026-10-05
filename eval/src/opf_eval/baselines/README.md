# Saved baselines

Raw detector results the notebooks load instead of running the detectors, so
the default session can be scored on a CPU-only runtime. One directory per
(dataset, seed); see [`baseline.py`](../baseline.py) for the format.

A baseline is used only when it covers every record in the sample with an
identical text hash and the same detector options; otherwise the notebook runs
the detector as usual. No fixture text is stored, only per-record hashes and
the detectors' spans.

To (re)build the default one, on a machine with a GPU:

```sh
python -m opf_eval.baseline build --n 200
```

It samples 200 records (enough for notebook 04; the 100-record default is a
prefix of the same sample), runs the default detectors and writes
`pii_masking_200k_s42/` here. Commit the directory.
