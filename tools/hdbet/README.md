# hd-bet-wrapper

HD-BET coupled with flexible file exploration for efficient batch processing.

A thin, configurable wrapper around [HD-BET](https://github.com/MIC-DKFZ/HD-BET)
that uses [nifti-finder](https://github.com/pkoutsouvelis/nifti-finder) to
discover inputs from arbitrarily nested neuroimaging datasets (including BIDS),
then drives HD-BET's batch mode while preserving the source directory layout
in the output tree.

## Why

- **Decouple discovery from inference.** Filter, glob, and compose the input
  set with nifti-finder; let HD-BET focus on inference.
- **Keep HD-BET's batching speedup.** Files are forwarded to HD-BET's
  `predict_from_files` so the model is initialized once for the entire batch,
  no matter where the inputs live.
- **Configuration as data.** One YAML file fully describes a run: inputs,
  filters, HD-BET flags, output layout, and logging.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../../core
pip install -e .
# with test extras
pip install -e .[test]
```

## Quickstart

```bash
hd-bet-wrapper -c configs/example.yaml
# preview only
hd-bet-wrapper -c configs/example.yaml --dry-run
```

See `configs/example.yaml` for the full schema. Briefly:

```yaml
input:
  dataset:
    root: /data/my_bids
    patterns: "**/anat/*T1w.nii*"
    filters:
      name: ExcludeFileSuffix
      kwargs: { suffix: "_bet" }

output:
  output_dir: /data/derivatives/hd-bet
  save_mask: true
  save_bet_image: true
  mask_suffix: mask          # -> <stem>_mask.nii.gz
  bet_image_suffix: bet      # -> <stem>_bet.nii.gz
  overwrite: false

hdbet:
  device: auto               # cuda | cpu | mps | auto
  use_tta: true
  num_processes_preprocessing: 4
  num_processes_segmentation_export: 8
  verbose: false

logging:
  log_dir: /data/derivatives/hd-bet/_logs
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

## Python use

```python
from hdbet_wrapper import HDBETRunner

runner = HDBETRunner(
    data={
        "root": "/data/my_bids",
        "patterns": "**/anat/*T1w.nii*",
        "filters": {"name": "ExcludeFileSuffix", "kwargs": {"suffix": "_bet"}},
    },
    output_dir="/data/derivatives/hd-bet",
    log_dir="/data/derivatives/hd-bet/_logs",
)
runner(device="auto", use_tta=True)
```

## Licensing and citations

This wrapper uses HD-BET. Please check its [license](https://github.com/MIC-DKFZ/HD-BET/blob/master/LICENSE).

If you use HD-BET in your research, the authors ask you cite the following publication:

```text
Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A,
Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P.
Automated brain extraction of multi-sequence MRI using artificial neural
networks. Hum Brain Mapp. 2019; 1-13. https://doi.org/10.1002/hbm.24750
```
