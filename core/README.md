# cliwrap-core

Shared building blocks for the wrappers in [`tools/`](../tools/).

## What's inside

- `cliwrap_core.BaseRunner` + `OutputSpec` — input resolution (single file,
  list, or `nifti-finder` dataset mapping) plus output-path planning with
  dataset-root mirroring and overwrite semantics. Tool runners subclass this
  and only have to write `__call__`.
- `cliwrap_core.config` — `load_yaml`, `parse_input`, `parse_logging`, and the
  `reject_unknown_keys` / `require_keys` helpers used by every tool's config
  loader.
- `cliwrap_core.explorer_factory.get_data_explorer` — factory that builds a
  `nifti_finder.FileFinder` from `patterns`, optional `levels`, and the nested
  YAML filter spec (supports `ComposeFilter` recursively).
- `cliwrap_core.logging_utils.setup_logger` — stdout + optional timestamped
  file handler, parametrised by logger name, bracketed label, and file prefix
  so each tool's logs are visibly attributable.
- `cliwrap_core.utils.resolve_path` / `get_ext`.

## Install

```bash
pip install -e .
# with test extras
pip install -e .[test]
pytest
```

## Build a new tool on top

```python
from cliwrap_core import BaseRunner, OutputSpec

class MyToolRunner(BaseRunner):
    def __init__(self, data, output_dir=None, my_suffix="mytool", overwrite=True,
                 log_dir=None, log_level="INFO"):
        self._my_suffix = my_suffix
        super().__init__(
            data, output_dir=output_dir, overwrite=overwrite,
            log_dir=log_dir, log_level=log_level,
            logger_name="mytool_wrapper", logger_label="MYTOOL WRAPPER",
            log_file_prefix="mytool_run",
        )

    def __call__(self, *, dry_run=False, **kwargs):
        plans = self._plan_outputs([OutputSpec("out", self._my_suffix, required=True)])
        if dry_run:
            for i, o in zip(plans["__inputs__"], plans["out"]):
                self._logger.info("%s -> %s", i, o)
            return
        # ... call your tool ...
```

See `tools/hdbet/src/hdbet_wrapper/runner.py` and
`tools/synthseg/src/synthseg_wrapper/runner.py` for full examples.
