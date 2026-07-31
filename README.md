# cli-wrappers

Configurable Python wrappers around neuroimaging CLI tools, with shared file
discovery via [nifti-finder](https://github.com/pkoutsouvelis/nifti-finder).

Each tool lives in its own folder under `tools/` with an independent
`pyproject.toml`, virtualenv, runner, config schema, and tests. The shared
machinery (input resolution, output planning, logging, YAML config helpers)
lives in `core/cliwrap-core` and is depended on by every tool.

## Layout

```text
cli-wrappers/
├── core/                     # cliwrap-core: shared discovery + base runner
└── tools/
    ├── hdbet/                # hd-bet-wrapper  (HD-BET 2.0.1 + nnUNet)
    └── synthseg/             # synthseg-wrapper (Photo-SynthSeg, TF 2.15)
```

Tools are installed independently — heavy dependencies (PyTorch for HD-BET,
TensorFlow 2.15 for SynthSeg) never share a venv.

## Install a tool

```bash
# Pick a tool, create a venv for it, install core + tool in editable mode.
python -m venv tools/hdbet/.venv
source tools/hdbet/.venv/bin/activate
pip install -e ./core
pip install -e ./tools/hdbet
```

```bash
python -m venv tools/synthseg/.venv
source tools/synthseg/.venv/bin/activate
pip install -e ./core
pip install -e "./tools/synthseg[gpu]"   # or [cpu] on macOS
# Provision Photo-SynthSeg + weights (one-time):
synthseg-setup --synthseg-home ~/Photo-SynthSeg --weights ./SynthSeg_models.zip
```

## Run a tool

Both tools take a single YAML config and an optional `--dry-run` flag:

```bash
hd-bet-wrapper   -c tools/hdbet/configs/example.yaml
synthseg-wrapper -c tools/synthseg/configs/example.yaml --dry-run
```

See each tool's `README.md` for its config schema and supported flags.

## Tests

Each tool's tests run independently:

```bash
cd tools/hdbet   && pytest    # 36 tests
cd tools/synthseg && pytest   # 27 tests
cd core           && pytest   # 63 tests
```

Or run them together from the repo root:

```bash
pytest --import-mode=importlib core/tests tools/hdbet/tests tools/synthseg/tests
```

## Adding a new tool

1. Create `tools/<name>/` with its own `pyproject.toml`, `src/<pkg>/`, and `tests/`.
2. Subclass `cliwrap_core.BaseRunner` and implement `__call__`.
3. Reuse `cliwrap_core.config.parse_input` and `parse_logging`; write your own
   `output` and `<tool>` section parsers.
4. Register a console script in your tool's `pyproject.toml`.

See `tools/hdbet/` and `tools/synthseg/` for working examples.
