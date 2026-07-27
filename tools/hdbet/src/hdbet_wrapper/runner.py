"""Top-level orchestration for an HD-BET wrapper run."""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import apply_bet, get_hdbet_predictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from cliwrap_core import BaseRunner, LogLevel, OutputSpec
from cliwrap_core.base_runner import InputData, PartIdx
from hdbet_wrapper.utils import print_hd_bet_citation

Device: TypeAlias = Literal["cuda", "cpu", "mps", "auto"]


class HDBETRunner(BaseRunner):
    """Orchestrates file discovery, HD-BET inference, and output saving.

    Args:
        data: Input spec; see :class:`cliwrap_core.BaseRunner` for the full
            list of accepted shapes (single file, list, or dataset mapping).
        output_dir: Root output directory; when ``None`` outputs land alongside
            inputs (or, in dataset mode, under the dataset root).
        save_mask: Whether to save the boolean brain masks. Defaults to True.
        save_bet_image: Whether to apply skull-stripping using the brain mask
            and save the output. Defaults to True.
        mask_suffix: Suffix appended to the input stem for the mask file.
        bet_image_suffix: Suffix appended to the input stem for the BET image.
        overwrite: Whether to overwrite existing outputs. Defaults to True.
        log_dir: Directory for a timestamped run log; stdout-only when None.
        log_level: Minimum log level. Defaults to ``"INFO"``.
    """

    def __init__(
        self,
        data: InputData,
        output_dir: Path | str | None = None,
        save_mask: bool = True,
        save_bet_image: bool = True,
        mask_suffix: str = "mask",
        bet_image_suffix: str = "bet",
        overwrite: bool = True,
        log_dir: Path | str | None = None,
        log_level: LogLevel = "INFO",
    ) -> None:
        if not isinstance(save_mask, bool):
            raise ValueError(
                f"save_mask must be a boolean, got {type(save_mask).__name__}"
            )
        if not isinstance(save_bet_image, bool):
            raise ValueError(
                f"save_bet_image must be a boolean, got {type(save_bet_image).__name__}"
            )
        if not save_mask and not save_bet_image:
            raise ValueError(
                "At least one of `save_mask` or `save_bet_image` must be True"
            )
        if not isinstance(mask_suffix, str):
            raise ValueError(
                f"mask_suffix must be a string, got {type(mask_suffix).__name__}"
            )
        if not isinstance(bet_image_suffix, str):
            raise ValueError(
                f"bet_image_suffix must be a string, got {type(bet_image_suffix).__name__}"
            )

        self._save_mask = save_mask
        self._save_bet_image = save_bet_image
        self._mask_suffix = mask_suffix
        self._bet_image_suffix = bet_image_suffix

        super().__init__(
            data=data,
            output_dir=output_dir,
            overwrite=overwrite,
            log_dir=log_dir,
            log_level=log_level,
            logger_name="hdbet_wrapper",
            logger_label="HD-BET WRAPPER",
            log_file_prefix="hdbet_run",
        )

    def _plan(self) -> dict[str, list[str]]:
        # HD-BET treats mask as always-produced (required) and bet as optional.
        # When save_bet_image is True, an existing bet output should also block
        # re-processing (parity with the original implementation).
        specs = [
            OutputSpec("mask", self._mask_suffix, save=True, required=True),
            OutputSpec(
                "bet",
                self._bet_image_suffix,
                save=self._save_bet_image,
                required=self._save_bet_image,
            ),
        ]
        return self._plan_outputs(specs)

    def _get_predictor(
        self,
        device: Device,
        use_tta: bool,
        verbose: bool,
    ) -> nnUNetPredictor:
        if device not in ("cuda", "cpu", "mps", "auto"):
            raise ValueError(
                f"Invalid device: {device!r}; must be one of 'cuda', 'cpu', 'mps', or 'auto'"
            )
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        maybe_download_parameters()

        return get_hdbet_predictor(
            use_tta=use_tta,
            device=torch.device(device),
            verbose=verbose,
        )

    def _apply_masks(
        self,
        inputs: list[list[str]],
        outputs_mask: list[str],
        outputs_bet: list[str],
        num_processes: int,
    ) -> None:
        res = []
        with Pool(num_processes) as p:
            for im, bet, out in zip(inputs, outputs_mask, outputs_bet):
                res.append(p.starmap_async(apply_bet, ((im[0], bet, out),)))
            [i.get() for i in res]

    def _clean_up(self, outputs_mask: list[str]) -> None:
        base_dir = Path(outputs_mask[0]).parent
        (base_dir / "dataset.json").unlink(missing_ok=True)
        (base_dir / "plans.json").unlink(missing_ok=True)
        (base_dir / "predict_from_raw_data_args.json").unlink(missing_ok=True)

        if not self._save_mask:
            [Path(i).unlink(missing_ok=True) for i in outputs_mask]

    def __call__(
        self,
        *,
        use_tta: bool = True,
        device: Device = "auto",
        num_processes_preprocessing: int = 4,
        num_processes_segmentation_export: int = 8,
        verbose: bool = False,
        dry_run: bool = False,
        num_parts: int = 1,
        part_idx: PartIdx = 0,
    ) -> None:
        """Run the HD-BET wrapper.

        Args:
            use_tta: Whether to use test-time augmentation. Defaults to True.
            device: ``"cuda" | "cpu" | "mps" | "auto"``.
            num_processes_preprocessing: HD-BET preprocessing parallelism.
            num_processes_segmentation_export: HD-BET export parallelism.
            verbose: Verbose flag forwarded to HD-BET.
            dry_run: Print planned input/output pairs without running HD-BET.
            num_parts: Split planned pairs into this many contiguous jobs.
                Defaults to ``1`` (no split). Independent of nnU-Net's
                ``num_parts`` (always ``1``).
            part_idx: Which part(s) to run, as an ``int`` or sequence of ints in
                ``[0, num_parts)``. Defaults to ``0``.
        """
        plans = self._plan()

        if len(plans["__inputs__"]) == 0:
            self._logger.info("No inputs to process.")
            return

        if len(plans["__inputs__"]) != len(self._input_files):
            self._logger.info(
                "Skipping %d files due to existing outputs and `overwrite=False`.",
                len(self._input_files) - len(plans["__inputs__"]),
            )

        plans = self._slice_plan(plans, num_parts=num_parts, part_idx=part_idx)
        if len(plans["__inputs__"]) == 0:
            self._logger.info("No inputs to process after part slicing.")
            return

        inputs = [[s] for s in plans["__inputs__"]]
        outputs_mask = plans["mask"]
        outputs_bet = plans["bet"] if self._save_bet_image else []

        if dry_run:
            for i in range(len(inputs)):
                self._logger.info(
                    "%s -> %s -> %s",
                    inputs[i][0],
                    outputs_mask[i] if self._save_mask else "N/A",
                    outputs_bet[i] if self._save_bet_image else "N/A",
                )
            return

        self._logger.info("Loading nnU-Net predictor...")
        predictor = self._get_predictor(device=device, use_tta=use_tta, verbose=verbose)
        self._logger.info("nnU-Net predictor loaded successfully.")

        self._logger.info("Running HD-BET...")
        print_hd_bet_citation(emit=self._logger.info)
        predictor.predict_from_files(
            inputs,
            outputs_mask,
            save_probabilities=False,
            overwrite=self._overwrite,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
            num_parts=1,
            part_id=0,
        )
        self._logger.info("HD-BET completed successfully.")

        if self._save_bet_image:
            self._logger.info("Applying brain masks...")
            self._apply_masks(
                inputs,
                outputs_mask,
                outputs_bet,
                num_processes=num_processes_preprocessing,
            )
            self._logger.info("Brain masks applied successfully.")

        self._clean_up(outputs_mask)

        self._logger.info("Done.")
