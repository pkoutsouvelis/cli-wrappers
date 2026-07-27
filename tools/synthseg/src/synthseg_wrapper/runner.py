"""Top-level orchestration for a SynthSeg wrapper run.

Photo-SynthSeg ships as a Python codebase, not a pip package; it's launched as
``python <home>/scripts/commands/SynthSeg_predict.py``. This runner discovers
inputs via ``nifti-finder``, plans output paths per input, then invokes that
script **once per run** in *batch mode* by passing list files to ``--i`` and
``--o``. That mirrors HD-BET's batched inference: the TF model is loaded a
single time for the entire run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cliwrap_core import BaseRunner, LogLevel, OutputSpec
from cliwrap_core.base_runner import InputData
from cliwrap_core.utils import resolve_path


class SynthSegRunner(BaseRunner):
    """Orchestrates discovery, batch invocation of SynthSeg_predict.py, and output saving.

    Args:
        data: Input spec (single file, list, or dataset mapping). See
            :class:`cliwrap_core.BaseRunner` for accepted shapes.
        output_dir: Root output directory. Required when ``save_volumes`` or
            ``save_qc`` is True (per-input CSV files are mirrored here).
        synthseg_home: Path to a Photo-SynthSeg clone with ``models/``
            populated. The predict script lives at
            ``{synthseg_home}/scripts/commands/SynthSeg_predict.py``.
        python_executable: Path to a Python interpreter with TF 2.15 +
            SynthSeg's deps. Defaults to ``sys.executable`` (i.e. the same
            interpreter running this wrapper).
        save_segmentation: Whether to keep the segmentation NIfTIs. Defaults to True.
        segmentation_suffix: Suffix for segmentation files.
        save_posteriors: Whether to save posterior probability NIfTIs.
        posteriors_suffix: Suffix for posteriors files.
        save_resampled: Whether to save the resampled images SynthSeg produces.
        resampled_suffix: Suffix for resampled files.
        save_volumes: Whether to ask SynthSeg for per-input volumes CSVs.
        volumes_suffix: Suffix for volumes CSV files (e.g. ``volumes`` ->
            ``<input_stem>_volumes.csv``).
        save_qc: Whether to ask SynthSeg for per-input QC CSVs.
        qc_suffix: Suffix for QC CSV files (e.g. ``qc`` ->
            ``<input_stem>_qc.csv``).
        overwrite: Whether to re-process inputs whose segmentation already exists.
        log_dir: Directory for a timestamped run log; stdout-only when None.
        log_level: Minimum log level.
    """

    def __init__(
        self,
        data: InputData,
        output_dir: Path | str | None = None,
        *,
        synthseg_home: Path | str,
        python_executable: Path | str | None = None,
        save_segmentation: bool = True,
        segmentation_suffix: str = "synthseg",
        save_posteriors: bool = False,
        posteriors_suffix: str = "post",
        save_resampled: bool = False,
        resampled_suffix: str = "resampled",
        save_volumes: bool = False,
        volumes_suffix: str = "volumes",
        save_qc: bool = False,
        qc_suffix: str = "qc",
        overwrite: bool = True,
        log_dir: Path | str | None = None,
        log_level: LogLevel = "INFO",
    ) -> None:
        for name, val in [
            ("save_segmentation", save_segmentation),
            ("save_posteriors", save_posteriors),
            ("save_resampled", save_resampled),
            ("save_volumes", save_volumes),
            ("save_qc", save_qc),
        ]:
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be a boolean, got {type(val).__name__}")
        if not save_segmentation:
            raise ValueError(
                "save_segmentation must be True; SynthSeg cannot run without it "
                "(other outputs are derived from the segmentation pass)."
            )
        for name, val in [
            ("segmentation_suffix", segmentation_suffix),
            ("posteriors_suffix", posteriors_suffix),
            ("resampled_suffix", resampled_suffix),
            ("volumes_suffix", volumes_suffix),
            ("qc_suffix", qc_suffix),
        ]:
            if not isinstance(val, str):
                raise ValueError(f"{name} must be a string, got {type(val).__name__}")

        self._save_segmentation = save_segmentation
        self._segmentation_suffix = segmentation_suffix
        self._save_posteriors = save_posteriors
        self._posteriors_suffix = posteriors_suffix
        self._save_resampled = save_resampled
        self._resampled_suffix = resampled_suffix
        self._save_volumes = save_volumes
        self._volumes_suffix = volumes_suffix
        self._save_qc = save_qc
        self._qc_suffix = qc_suffix

        self._synthseg_home = resolve_path(synthseg_home)
        self._predict_script = (
            self._synthseg_home / "scripts" / "commands" / "SynthSeg_predict.py"
        )
        if not self._predict_script.exists():
            raise FileNotFoundError(
                f"SynthSeg predict script not found at {self._predict_script}. "
                "Did you run `synthseg-setup` to clone Photo-SynthSeg?"
            )

        self._python = (
            str(resolve_path(python_executable))
            if python_executable is not None
            else sys.executable
        )

        super().__init__(
            data=data,
            output_dir=output_dir,
            overwrite=overwrite,
            log_dir=log_dir,
            log_level=log_level,
            logger_name="synthseg_wrapper",
            logger_label="SYNTHSEG WRAPPER",
            log_file_prefix="synthseg_run",
        )

        if (self._save_volumes or self._save_qc) and self._output_dir is None:
            raise ValueError(
                "output_dir is required when save_volumes or save_qc is True."
            )

    def _plan(self) -> dict[str, list[str]]:
        specs = [
            OutputSpec(
                "segmentation",
                self._segmentation_suffix,
                save=True,
                required=True,
            ),
            OutputSpec(
                "posteriors",
                self._posteriors_suffix,
                save=self._save_posteriors,
                required=self._save_posteriors,
            ),
            OutputSpec(
                "resampled",
                self._resampled_suffix,
                save=self._save_resampled,
                required=self._save_resampled,
            ),
        ]
        plans = self._plan_outputs(specs)
        inputs = plans["__inputs__"]
        if self._save_volumes:
            plans["volumes"] = [
                str(self._csv_output_path(Path(inp), self._volumes_suffix))
                for inp in inputs
            ]
        if self._save_qc:
            plans["qc"] = [
                str(self._csv_output_path(Path(inp), self._qc_suffix)) for inp in inputs
            ]
        return plans

    def _csv_output_path(self, p: Path, suffix: str) -> Path:
        """Per-input CSV path for batch-mode ``--vol`` / ``--qc`` list files."""
        stem, _ = self._stem_and_ext(p)
        out = self._out_parent_for(p) / f"{stem}_{suffix}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def _write_list(self, parent: Path, filename: str, items: list[str]) -> Path:
        p = parent / filename
        p.write_text("\n".join(items) + "\n")
        return p

    def _build_command(
        self,
        list_dir: Path,
        plans: dict[str, list[str]],
        *,
        parc: bool,
        robust: bool,
        fast: bool,
        ct: bool,
        cpu: bool,
        threads: int,
        crop: list[int] | None,
        v1: bool,
    ) -> list[str]:
        inputs_list = self._write_list(list_dir, "inputs.txt", plans["__inputs__"])
        seg_list = self._write_list(
            list_dir, "segmentations.txt", plans["segmentation"]
        )

        cmd: list[str] = [
            self._python,
            str(self._predict_script),
            "--i",
            str(inputs_list),
            "--o",
            str(seg_list),
            "--threads",
            str(threads),
        ]

        if self._save_posteriors:
            post_list = self._write_list(
                list_dir, "posteriors.txt", plans["posteriors"]
            )
            cmd += ["--post", str(post_list)]
        if self._save_resampled:
            res_list = self._write_list(list_dir, "resampled.txt", plans["resampled"])
            cmd += ["--resample", str(res_list)]
        if self._save_volumes:
            vol_list = self._write_list(list_dir, "volumes.txt", plans["volumes"])
            cmd += ["--vol", str(vol_list)]
        if self._save_qc:
            qc_list = self._write_list(list_dir, "qc.txt", plans["qc"])
            cmd += ["--qc", str(qc_list)]

        for flag, on in [
            ("parc", parc),
            ("robust", robust),
            ("fast", fast),
            ("ct", ct),
            ("cpu", cpu),
            ("v1", v1),
        ]:
            if on:
                cmd.append(f"--{flag}")
        if crop:
            cmd += ["--crop", *map(str, crop)]

        return cmd

    def _log_plan(self, plans: dict[str, list[str]]) -> None:
        for i, in_path in enumerate(plans["__inputs__"]):
            seg = plans["segmentation"][i] if self._save_segmentation else "N/A"
            post = (
                plans["posteriors"][i]
                if self._save_posteriors and plans["posteriors"]
                else "N/A"
            )
            res = (
                plans["resampled"][i]
                if self._save_resampled and plans["resampled"]
                else "N/A"
            )
            self._logger.info("%s -> %s | post=%s | res=%s", in_path, seg, post, res)

    def _run_subprocess(self, cmd: list[str]) -> None:
        """Invoke the SynthSeg predict script. Override in tests if needed."""
        env = os.environ.copy()
        subprocess.run(cmd, check=True, env=env)

    def __call__(
        self,
        *,
        parc: bool = False,
        robust: bool = False,
        fast: bool = False,
        ct: bool = False,
        cpu: bool = False,
        threads: int = 1,
        crop: list[int] | None = None,
        v1: bool = False,
        dry_run: bool = False,
        num_parts: int = 1,
        part_idx: int | list[int] = 0,
    ) -> None:
        """Run the SynthSeg wrapper.

        Args:
            parc: Whether to perform cortex parcellation (``--parc``).
            robust: Use robust predictions (``--robust``); implies ``fast=True``.
            fast: Bypass some postprocessing for faster predictions (``--fast``).
            ct: Clip intensities to [0, 80] for CT scans (``--ct``).
            cpu: Force CPU even if CUDA is available (``--cpu``).
            threads: Number of CPU threads (``--threads``).
            crop: Optional list of patch sizes (``--crop X Y Z``).
            v1: Use SynthSeg 1.0 instead of 2.0 (``--v1``).
            dry_run: Plan inputs/outputs and log them without running SynthSeg.
            num_parts: Split planned pairs into this many contiguous jobs.
                Defaults to ``1`` (no split).
            part_idx: Which part(s) to run, as an ``int`` or ``list[int]`` in
                ``[0, num_parts)``. Defaults to ``0``.
        """
        plans = self._plan()
        if not plans["__inputs__"]:
            self._logger.info("No inputs to process.")
            return

        if len(plans["__inputs__"]) != len(self._input_files):
            self._logger.info(
                "Skipping %d files due to existing outputs and `overwrite=False`.",
                len(self._input_files) - len(plans["__inputs__"]),
            )

        plans = self._slice_plan(plans, num_parts=num_parts, part_idx=part_idx)
        if not plans["__inputs__"]:
            self._logger.info("No inputs to process after part slicing.")
            return

        if dry_run:
            self._log_plan(plans)
            return

        with tempfile.TemporaryDirectory(prefix="synthseg_lists_") as td:
            cmd = self._build_command(
                Path(td),
                plans,
                parc=parc,
                robust=robust,
                fast=fast,
                ct=ct,
                cpu=cpu,
                threads=threads,
                crop=crop,
                v1=v1,
            )
            self._logger.info("Running SynthSeg: %s", " ".join(cmd))
            self._run_subprocess(cmd)

        # If the user asked us NOT to keep the segmentation, delete it. The
        # predict script always writes segs, so this is post-hoc cleanup.
        if not self._save_segmentation:  # pragma: no cover - guarded above
            for p in plans["segmentation"]:
                Path(p).unlink(missing_ok=True)

        self._logger.info("Done.")
