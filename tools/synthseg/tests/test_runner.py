"""Tests for :class:`synthseg_wrapper.runner.SynthSegRunner`."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthseg_wrapper.runner import SynthSegRunner

# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_synthseg_home_must_contain_predict_script(tmp_path: Path):
    p = tmp_path / "x.nii.gz"
    p.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="predict script not found"):
        SynthSegRunner(data=p, synthseg_home=tmp_path / "missing")


def test_save_segmentation_false_rejected(
    single_nifti: Path, fake_photo_synthseg: Path
):
    with pytest.raises(ValueError, match="save_segmentation must be True"):
        SynthSegRunner(
            data=single_nifti,
            synthseg_home=fake_photo_synthseg,
            save_segmentation=False,
        )


def test_volumes_requires_output_dir(single_nifti: Path, fake_photo_synthseg: Path):
    with pytest.raises(ValueError, match="output_dir is required"):
        SynthSegRunner(
            data=single_nifti,
            synthseg_home=fake_photo_synthseg,
            save_volumes=True,
        )


def test_invalid_data_type(fake_photo_synthseg: Path):
    with pytest.raises(ValueError, match="data must be"):
        SynthSegRunner(data=42, synthseg_home=fake_photo_synthseg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Plan / dry-run
# ---------------------------------------------------------------------------


def test_dry_run_logs_plan_and_skips_subprocess(
    single_nifti: Path, fake_photo_synthseg: Path, patch_synthseg, capsys
):
    runner = SynthSegRunner(
        data=single_nifti,
        synthseg_home=fake_photo_synthseg,
    )
    runner(dry_run=True)
    out = capsys.readouterr().out
    assert "scan_T1w_synthseg.nii.gz" in out
    assert patch_synthseg.calls == []


def test_dataset_mode_mirrors_under_root(
    bids_tree: Path, fake_photo_synthseg: Path, patch_synthseg, capsys
):
    runner = SynthSegRunner(
        data={
            "root": str(bids_tree),
            "patterns": "**/anat/*T1w.nii*",
        },
        synthseg_home=fake_photo_synthseg,
    )
    runner(dry_run=True)
    out = capsys.readouterr().out

    expected = (
        bids_tree / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w_synthseg.nii.gz"
    )
    assert str(expected) in out


# ---------------------------------------------------------------------------
# Full execution path (subprocess faked)
# ---------------------------------------------------------------------------


def test_full_run_writes_segmentation(
    single_nifti: Path, fake_photo_synthseg: Path, patch_synthseg
):
    runner = SynthSegRunner(
        data=single_nifti,
        synthseg_home=fake_photo_synthseg,
    )
    runner()

    seg = single_nifti.parent / "scan_T1w_synthseg.nii.gz"
    assert seg.exists()
    assert len(patch_synthseg.calls) == 1


def test_run_passes_optional_flags(
    single_nifti: Path, fake_photo_synthseg: Path, patch_synthseg
):
    runner = SynthSegRunner(
        data=single_nifti,
        synthseg_home=fake_photo_synthseg,
        save_posteriors=True,
        save_resampled=True,
    )
    runner(parc=True, fast=True, cpu=True, threads=4, crop=[192, 192, 192])

    [cmd] = patch_synthseg.calls
    assert "--parc" in cmd
    assert "--fast" in cmd
    assert "--cpu" in cmd
    assert "--threads" in cmd and cmd[cmd.index("--threads") + 1] == "4"
    crop_idx = cmd.index("--crop")
    assert cmd[crop_idx + 1 : crop_idx + 4] == ["192", "192", "192"]
    assert "--post" in cmd
    assert "--resample" in cmd

    assert (single_nifti.parent / "scan_T1w_post.nii.gz").exists()
    assert (single_nifti.parent / "scan_T1w_resampled.nii.gz").exists()


def test_run_writes_volumes_and_qc_csvs(
    bids_tree: Path, tmp_path: Path, fake_photo_synthseg: Path, patch_synthseg
):
    out_dir = tmp_path / "outs"
    runner = SynthSegRunner(
        data={"root": str(bids_tree), "patterns": "**/anat/*T1w.nii*"},
        output_dir=out_dir,
        synthseg_home=fake_photo_synthseg,
        save_volumes=True,
        save_qc=True,
        volumes_suffix="vols",
        qc_suffix="qc-scores",
    )
    runner()

    for sub in ("sub-01", "sub-02"):
        anat = out_dir / sub / "ses-01" / "anat"
        assert (anat / f"{sub}_ses-01_T1w_vols.csv").exists()
        assert (anat / f"{sub}_ses-01_T1w_qc-scores.csv").exists()
    [cmd] = patch_synthseg.calls
    assert "--vol" in cmd
    assert "--qc" in cmd
    assert len(patch_synthseg.volumes[0]) == 2
    assert len(patch_synthseg.qc[0]) == 2


def test_overwrite_false_skips_existing_outputs(
    single_nifti: Path, fake_photo_synthseg: Path, patch_synthseg
):
    existing = single_nifti.parent / "scan_T1w_synthseg.nii.gz"
    existing.write_bytes(b"sentinel")
    runner = SynthSegRunner(
        data=single_nifti,
        synthseg_home=fake_photo_synthseg,
        overwrite=False,
    )
    runner()

    assert existing.read_bytes() == b"sentinel"
    assert patch_synthseg.calls == []


def test_run_writes_log_file(
    single_nifti: Path, fake_photo_synthseg: Path, patch_synthseg, tmp_path: Path
):
    log_dir = tmp_path / "logs"
    runner = SynthSegRunner(
        data=single_nifti,
        synthseg_home=fake_photo_synthseg,
        log_dir=log_dir,
    )
    runner()

    [log_file] = log_dir.glob("synthseg_run_*.log")
    contents = log_file.read_text()
    assert "Running SynthSeg" in contents
    assert "Done." in contents
    assert "[SYNTHSEG WRAPPER]" in contents


def test_batches_in_single_subprocess_call(
    bids_tree: Path, fake_photo_synthseg: Path, patch_synthseg
):
    runner = SynthSegRunner(
        data={"root": str(bids_tree), "patterns": "**/anat/*T1w.nii*"},
        synthseg_home=fake_photo_synthseg,
    )
    runner()

    # Single SynthSeg invocation regardless of N inputs (matches HD-BET batching).
    assert len(patch_synthseg.calls) == 1
    assert len(patch_synthseg.inputs[0]) == 2
    assert len(patch_synthseg.segmentations[0]) == 2


def test_part_idx_filters_batch_lists(
    tmp_path: Path, fake_photo_synthseg: Path, patch_synthseg, capsys
):
    files = []
    for name in ("a.nii.gz", "b.nii.gz", "c.nii.gz", "d.nii.gz"):
        p = tmp_path / name
        p.write_bytes(b"")
        files.append(p)

    runner = SynthSegRunner(
        data=files,
        output_dir=tmp_path / "out",
        synthseg_home=fake_photo_synthseg,
    )
    runner(num_parts=2, part_idx=1)

    assert len(patch_synthseg.calls) == 1
    assert patch_synthseg.inputs[0] == [str(files[2]), str(files[3])]
    assert len(patch_synthseg.segmentations[0]) == 2


def test_part_idx_list_dry_run(
    tmp_path: Path, fake_photo_synthseg: Path, patch_synthseg, capsys
):
    files = []
    for name in ("a.nii.gz", "b.nii.gz", "c.nii.gz", "d.nii.gz"):
        p = tmp_path / name
        p.write_bytes(b"")
        files.append(p)

    runner = SynthSegRunner(
        data=files,
        output_dir=tmp_path / "out",
        synthseg_home=fake_photo_synthseg,
    )
    runner(num_parts=4, part_idx=[0, 2], dry_run=True)
    out = capsys.readouterr().out
    assert str(files[0]) in out
    assert str(files[2]) in out
    assert str(files[1]) not in out
    assert "Selecting part(s) of 4 discovered input(s)" in out
    assert patch_synthseg.calls == []
