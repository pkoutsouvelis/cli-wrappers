"""Tests for :class:`hdbet_wrapper.runner.HDBETRunner`."""

from __future__ import annotations

from pathlib import Path

import pytest

from hdbet_wrapper.runner import HDBETRunner

# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_invalid_data_type():
    with pytest.raises(ValueError, match="data must be"):
        HDBETRunner(data=42)  # type: ignore[arg-type]


def test_save_both_false(tmp_path: Path):
    p = tmp_path / "x.nii.gz"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="At least one of"):
        HDBETRunner(data=p, save_mask=False, save_bet_image=False)


def test_bad_log_level(single_nifti: Path):
    with pytest.raises(ValueError, match="log_level"):
        HDBETRunner(data=single_nifti, log_level="VERBOSE")  # type: ignore[arg-type]


def test_list_must_contain_paths_or_strings(tmp_path: Path):
    with pytest.raises(ValueError, match="Path or str"):
        HDBETRunner(data=[1, 2])  # type: ignore[list-item]


def test_dataset_requires_root_and_patterns(tmp_path: Path):
    with pytest.raises(ValueError, match="`root` key"):
        HDBETRunner(data={"patterns": "*.nii.gz"})

    with pytest.raises(ValueError, match="`patterns` key"):
        HDBETRunner(data={"root": str(tmp_path)})


# ---------------------------------------------------------------------------
# Output path planning (no HD-BET execution; runs via dry_run)
# ---------------------------------------------------------------------------


def test_single_file_writes_alongside_input(single_nifti: Path, patch_hdbet, capsys):
    runner = HDBETRunner(data=single_nifti)
    runner(dry_run=True)

    out = capsys.readouterr().out
    base = "scan_T1w"
    assert f"{base}_mask.nii.gz" in out
    assert f"{base}_bet.nii.gz" in out
    assert str(single_nifti) in out


def test_list_mode_uses_output_dir(tmp_path: Path, patch_hdbet, capsys):
    files = []
    for name in ("a.nii.gz", "b.nii.gz"):
        p = tmp_path / "src" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        files.append(p)

    out_dir = tmp_path / "outs"
    runner = HDBETRunner(data=files, output_dir=out_dir)
    runner(dry_run=True)

    out = capsys.readouterr().out
    assert str(out_dir / "a_mask.nii.gz") in out
    assert str(out_dir / "b_mask.nii.gz") in out


def test_dataset_mode_mirrors_under_root(bids_tree: Path, patch_hdbet, capsys):
    runner = HDBETRunner(
        data={
            "root": str(bids_tree),
            "patterns": "**/anat/*T1w.nii*",
            "filters": {
                "name": "ExcludeFileSuffix",
                "kwargs": {"suffix": "_bet"},
            },
        }
    )
    runner(dry_run=True)
    out = capsys.readouterr().out

    expected = (
        bids_tree / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w_mask.nii.gz"
    )
    assert str(expected) in out


def test_custom_suffixes(single_nifti: Path, patch_hdbet, capsys):
    runner = HDBETRunner(
        data=single_nifti,
        mask_suffix="brainmask",
        bet_image_suffix="stripped",
    )
    runner(dry_run=True)
    out = capsys.readouterr().out
    assert "scan_T1w_brainmask.nii.gz" in out
    assert "scan_T1w_stripped.nii.gz" in out


# ---------------------------------------------------------------------------
# Full execution path (HD-BET internals patched out)
# ---------------------------------------------------------------------------


def test_full_run_writes_mask_and_bet(single_nifti: Path, patch_hdbet):
    runner = HDBETRunner(data=single_nifti)
    runner()

    mask = single_nifti.parent / "scan_T1w_mask.nii.gz"
    bet = single_nifti.parent / "scan_T1w_bet.nii.gz"
    assert mask.exists()
    assert bet.exists()
    assert len(patch_hdbet.calls) == 1


def test_save_mask_false_removes_mask(single_nifti: Path, patch_hdbet):
    runner = HDBETRunner(data=single_nifti, save_mask=False)
    runner()

    assert not (single_nifti.parent / "scan_T1w_mask.nii.gz").exists()
    assert (single_nifti.parent / "scan_T1w_bet.nii.gz").exists()


def test_save_bet_false_keeps_only_mask(single_nifti: Path, patch_hdbet):
    runner = HDBETRunner(data=single_nifti, save_bet_image=False)
    runner()

    assert (single_nifti.parent / "scan_T1w_mask.nii.gz").exists()
    assert not (single_nifti.parent / "scan_T1w_bet.nii.gz").exists()


def test_overwrite_false_skips_existing_outputs(single_nifti: Path, patch_hdbet):
    existing_mask = single_nifti.parent / "scan_T1w_mask.nii.gz"
    existing_mask.write_bytes(b"sentinel")

    runner = HDBETRunner(data=single_nifti, overwrite=False)
    runner()

    assert existing_mask.read_bytes() == b"sentinel"
    assert patch_hdbet.calls == []


def test_dry_run_skips_predictor(single_nifti: Path, patch_hdbet):
    runner = HDBETRunner(data=single_nifti)
    runner(dry_run=True)

    assert patch_hdbet.calls == []
    assert not hasattr(patch_hdbet, "init_kwargs")


def test_run_writes_log_file(single_nifti: Path, patch_hdbet, tmp_path: Path):
    log_dir = tmp_path / "logs"
    runner = HDBETRunner(data=single_nifti, log_dir=log_dir)
    runner()

    [log_file] = log_dir.glob("hdbet_run_*.log")
    contents = log_file.read_text()
    assert "Loading nnU-Net predictor" in contents
    assert "HD-BET completed successfully" in contents
    assert "Done." in contents
    assert "HD-BET (MIC-DKFZ)" in contents


def test_device_auto_resolves_to_cpu_when_no_gpu(single_nifti: Path, patch_hdbet):
    runner = HDBETRunner(data=single_nifti)
    runner(device="auto")
    assert patch_hdbet.init_kwargs["device"] == "cpu"  # type: ignore[attr-defined]


def test_invalid_device_raises(single_nifti: Path, patch_hdbet):
    runner = HDBETRunner(data=single_nifti)
    with pytest.raises(ValueError, match="Invalid device"):
        runner(device="tpu")  # type: ignore[arg-type]


def test_part_idx_filters_predictor_inputs(tmp_path: Path, patch_hdbet):
    files = []
    for name in ("a.nii.gz", "b.nii.gz", "c.nii.gz", "d.nii.gz"):
        p = tmp_path / name
        p.write_bytes(b"")
        files.append(p)

    runner = HDBETRunner(data=files, output_dir=tmp_path / "out")
    runner(num_parts=2, part_idx=0)

    assert len(patch_hdbet.calls) == 1
    call = patch_hdbet.calls[0]
    assert [inp[0] for inp in call["inputs"]] == [str(files[0]), str(files[1])]
    assert call["num_parts"] == 1
    assert call["part_id"] == 0
    assert (tmp_path / "out" / "a_bet.nii.gz").exists()
    assert (tmp_path / "out" / "b_bet.nii.gz").exists()
    assert not (tmp_path / "out" / "c_bet.nii.gz").exists()


def test_part_idx_list_selects_multiple_parts(tmp_path: Path, patch_hdbet, capsys):
    files = []
    for name in ("a.nii.gz", "b.nii.gz", "c.nii.gz", "d.nii.gz"):
        p = tmp_path / name
        p.write_bytes(b"")
        files.append(p)

    runner = HDBETRunner(data=files, output_dir=tmp_path / "out")
    runner(num_parts=4, part_idx=[0, 2], dry_run=True)

    out = capsys.readouterr().out
    assert str(files[0]) in out
    assert str(files[2]) in out
    assert str(files[1]) not in out
    assert str(files[3]) not in out
    assert "before part slicing" in out
    assert "after part slicing" in out
    assert patch_hdbet.calls == []


def test_invalid_part_idx_raises(single_nifti: Path, patch_hdbet):
    runner = HDBETRunner(data=single_nifti)
    with pytest.raises(ValueError, match="part_idx"):
        runner(num_parts=2, part_idx=2)
