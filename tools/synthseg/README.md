# synthseg-wrapper

[Photo-SynthSeg](https://github.com/MGH-LEMoN/Photo-SynthSeg) (TF 2.15) coupled
with flexible file exploration for efficient batch processing.

A thin, configurable wrapper that uses [nifti-finder](https://github.com/pkoutsouvelis/nifti-finder)
to discover inputs from arbitrarily nested neuroimaging datasets (including
BIDS), then invokes the upstream `SynthSeg_predict.py` script **once** with
input/output list files so the TensorFlow model is loaded a single time per
run. Output layout mirrors the input dataset under `output_dir`.

## Why

- **Decouple discovery from inference.** Filter, glob, and compose the input
  set with nifti-finder; let SynthSeg focus on inference.
- **Keep batching cheap.** SynthSeg_predict.py is invoked exactly once per
  run; the TF model is loaded once regardless of the number of inputs.
- **No vendor lock-in to the wrapper venv.** SynthSeg can run in the same
  venv as this wrapper, or in a separate interpreter pointed at via
  `synthseg.python_executable`.
- **Configuration as data.** One YAML file fully describes a run: inputs,
  filters, output layout, SynthSeg flags, and logging.

## Install

SynthSeg's TF 2.15 pin and its CUDA bundle constrain Python to 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ../../core
# Pick ONE extras flavour:
pip install -e .[gpu]   # tensorflow[and-cuda]==2.15  (Linux GPU host)
pip install -e .[cpu]   # tensorflow==2.15            (CPU-only / macOS)
pip install -e .[test]  # pytest + nibabel, on top of the above
```

If you prefer to keep this wrapper's venv lean, install just the wrapper here
and create a **separate** venv that contains TF 2.15 + SynthSeg's deps; set
`synthseg.python_executable` in your config to that other venv's Python.

## One-time provisioning

The upstream Photo-SynthSeg fork is not pip-installable. `synthseg-setup`
mirrors the project's reference Dockerfile minus the OS-level steps:

```bash
# (1) Download SynthSeg_models.zip manually from SynthSeg's SharePoint, then:
synthseg-setup \
  --synthseg-home ~/Photo-SynthSeg \
  --weights ~/Downloads/SynthSeg_models.zip
```

What it does:

1. `git clone --branch synthseg_tf2.15 --single-branch https://github.com/MGH-LEMoN/Photo-SynthSeg.git ~/Photo-SynthSeg`
2. Unzips the weights archive into the clone
3. Renames `SynthSeg_models/` -> `models/` (matches the Dockerfile)
4. Strips `__MACOSX/` cruft

Re-run with `--skip-clone` to just (re)install weights, or `--force` to wipe
and re-clone.

## Quickstart

```bash
synthseg-wrapper -c configs/example.yaml
# preview only
synthseg-wrapper -c configs/example.yaml --dry-run
```

See `configs/example.yaml` for the full schema. Briefly:

```yaml
input:
  dataset:
    root: /data/my_bids
    patterns: "**/anat/*T1w.nii*"
    filters:
      name: ExcludeFileSuffix
      kwargs: { suffix: "_synthseg" }

output:
  output_dir: /data/derivatives/synthseg
  save_segmentation: true
  segmentation_suffix: synthseg
  save_posteriors: false
  posteriors_suffix: post
  save_resampled: false
  resampled_suffix: resampled
  save_volumes: true
  volumes_filename: volumes.csv
  save_qc: true
  qc_filename: qc.csv
  overwrite: false

synthseg:
  synthseg_home: ~/Photo-SynthSeg
  python_executable: null         # null -> use the wrapper's interpreter
  parc: false                     # --parc
  robust: false                   # --robust (implies fast)
  fast: false                     # --fast
  ct: false                       # --ct
  cpu: false                      # --cpu
  threads: 4                      # --threads N
  crop: null                      # --crop X [Y Z]
  v1: false                       # --v1

logging:
  log_dir: /data/derivatives/synthseg/_logs
  level: INFO
```

Explicit file-list mode swaps the input block (mutually exclusive with
`dataset`):

```yaml
input:
  files:
    - /data/sub-01/anat/sub-01_T1w.nii.gz
    - /data/sub-02/anat/sub-02_T1w.nii.gz
```

Output behaviour:

- `dataset` mode mirrors each file's `root`-relative path under `output_dir`
  (so BIDS layout is preserved). When `output_dir` is omitted, outputs land
  alongside the inputs.
- `files` mode writes outputs to `output_dir` (flat) when set, else alongside
  each input.
- `volumes_filename` and `qc_filename` (if enabled) are written at
  `output_dir/` and therefore require `output_dir` to be set.

## Python use

```python
from synthseg_wrapper import SynthSegRunner

runner = SynthSegRunner(
    data={
        "root": "/data/my_bids",
        "patterns": "**/anat/*T1w.nii*",
        "filters": {"name": "ExcludeFileSuffix", "kwargs": {"suffix": "_synthseg"}},
    },
    output_dir="/data/derivatives/synthseg",
    synthseg_home="/opt/Photo-SynthSeg",
    save_volumes=True,
    save_qc=True,
    log_dir="/data/derivatives/synthseg/_logs",
)
runner(parc=True, threads=4)
```

## Licensing and citations

This wrapper drives Photo-SynthSeg. Please check the upstream
[SynthSeg license](https://github.com/BBillot/SynthSeg/blob/master/LICENSE.md)
and [Photo-SynthSeg](https://github.com/MGH-LEMoN/Photo-SynthSeg).

If you use SynthSeg in your research, please cite (see SynthSeg's bibtex for
the full list):

```text
Billot B, et al. SynthSeg: Segmentation of brain MRI scans of any contrast
and resolution without retraining. Medical Image Analysis 86 (2023): 102789.
```
