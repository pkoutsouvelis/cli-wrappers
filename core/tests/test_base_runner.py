"""Tests for :class:`cliwrap_core.base_runner.BaseRunner`."""

from __future__ import annotations

from pathlib import Path

import pytest

from cliwrap_core.base_runner import BaseRunner, OutputSpec


class _DummyRunner(BaseRunner):
    """Minimal subclass exposing the internals for testing."""

    def __init__(self, **kwargs):
        kwargs.setdefault("logger_name", "cliwrap_dummy")
        kwargs.setdefault("logger_label", "DUMMY")
        super().__init__(**kwargs)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_invalid_data_type():
    with pytest.raises(ValueError, match="data must be"):
        _DummyRunner(data=42)  # type: ignore[arg-type]


def test_bad_log_level(tmp_path: Path):
    p = _touch(tmp_path / "x.nii.gz")
    with pytest.raises(ValueError, match="log_level"):
        _DummyRunner(data=p, log_level="VERBOSE")  # type: ignore[arg-type]


def test_list_must_contain_paths_or_strings():
    with pytest.raises(ValueError, match="Path or str"):
        _DummyRunner(data=[1, 2])  # type: ignore[list-item]


def test_dataset_requires_root_and_patterns(tmp_path: Path):
    with pytest.raises(ValueError, match="`root` key"):
        _DummyRunner(data={"patterns": "*.nii.gz"})

    with pytest.raises(ValueError, match="`patterns` key"):
        _DummyRunner(data={"root": str(tmp_path)})


def test_from_file_loads_paths(tmp_path: Path, capsys):
    a = _touch(tmp_path / "a.nii.gz")
    b = _touch(tmp_path / "b.nii.gz")
    list_file = tmp_path / "paths.txt"
    list_file.write_text(f"{a}\n# skip\n{b}\n", encoding="utf-8")

    runner = _DummyRunner(data={"from_file": str(list_file)})
    assert runner._input_files == [a.resolve(), b.resolve()]
    assert runner._root is None
    out = capsys.readouterr().out
    assert "Reading paths from:" in out
    assert "Loaded 2 path(s)" in out


def test_from_file_with_root_mirrors_outputs(tmp_path: Path):
    root = tmp_path / "FOMO300k"
    p = _touch(root / "PT001" / "sub-01" / "anat" / "T1w.nii.gz")
    list_file = tmp_path / "paths.txt"
    list_file.write_text(f"{p}\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    runner = _DummyRunner(
        data={"from_file": str(list_file), "root": str(root)},
        output_dir=out_dir,
    )
    assert runner._root == root.resolve()
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["seg"] == [
        str(out_dir / "PT001" / "sub-01" / "anat" / "T1w_synthseg.nii.gz")
    ]


def test_files_with_root_mirrors_outputs(tmp_path: Path):
    root = tmp_path / "ds"
    p = _touch(root / "PT001" / "anat" / "T1w.nii.gz")
    out_dir = tmp_path / "out"
    runner = _DummyRunner(
        data={"files": [str(p)], "root": str(root)},
        output_dir=out_dir,
    )
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["seg"] == [str(out_dir / "PT001" / "anat" / "T1w_synthseg.nii.gz")]


def test_explicit_paths_reject_outside_root(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    outside = _touch(tmp_path / "other" / "a.nii.gz")
    with pytest.raises(ValueError, match="not under root"):
        _DummyRunner(data={"files": [str(outside)], "root": str(root)})


def test_skip_resolve_still_validates_root(tmp_path: Path, capsys):
    """Flag False skips filepath resolve; under-root check still runs."""
    root = tmp_path / "ds"
    root.mkdir()
    p = root / "PT001" / "a.nii.gz"
    p.parent.mkdir(parents=True)
    list_file = tmp_path / "paths.txt"
    # Write resolved form so relative_to(root.resolve()) succeeds without resolve().
    list_file.write_text(f"{p.resolve()}\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    runner = _DummyRunner(
        data={"from_file": str(list_file), "root": str(root)},
        output_dir=out_dir,
        resolve_explicit_filepaths=False,
    )
    assert runner._root == root.resolve()
    assert runner._input_files == [p.resolve()]
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["seg"] == [str(out_dir / "PT001" / "a_synthseg.nii.gz")]
    out = capsys.readouterr().out
    assert "resolve skipped" in out
    assert "validation skipped" not in out


def test_skip_resolve_rejects_outside_root(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    outside = tmp_path / "other" / "a.nii.gz"
    outside.parent.mkdir(parents=True)
    list_file = tmp_path / "paths.txt"
    list_file.write_text(f"{outside}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not under root"):
        _DummyRunner(
            data={"from_file": str(list_file), "root": str(root)},
            resolve_explicit_filepaths=False,
        )


def test_from_file_rejects_patterns(tmp_path: Path):
    list_file = tmp_path / "paths.txt"
    list_file.write_text(f"{tmp_path / 'a.nii.gz'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="patterns"):
        _DummyRunner(
            data={"from_file": str(list_file), "root": str(tmp_path), "patterns": "*"}
        )


def test_plan_outputs_single_file(tmp_path: Path):
    p = _touch(tmp_path / "scan.nii.gz")
    runner = _DummyRunner(data=p)
    plans = runner._plan_outputs(
        [
            OutputSpec("seg", "synthseg", save=True, required=True),
            OutputSpec("post", "post", save=False),
        ]
    )
    assert plans["__inputs__"] == [str(p)]
    assert plans["seg"] == [str(tmp_path / "scan_synthseg.nii.gz")]
    assert plans["post"] == [str(tmp_path / "scan_post.nii.gz")]


def test_plan_outputs_mirrors_under_root(tmp_path: Path):
    root = tmp_path / "ds"
    p = _touch(root / "sub-01" / "anat" / "T1w.nii.gz")
    out_dir = tmp_path / "out"
    runner = _DummyRunner(
        data={"root": str(root), "patterns": "**/*.nii.gz"},
        output_dir=out_dir,
    )
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["__inputs__"] == [str(p)]
    assert plans["seg"] == [str(out_dir / "sub-01" / "anat" / "T1w_synthseg.nii.gz")]


def test_dataset_levels_are_forwarded(tmp_path: Path):
    root = tmp_path / "ds"
    keep = _touch(root / "PT001" / "a_T1w.nii.gz")
    _touch(root / "other" / "b_T1w.nii.gz")
    runner = _DummyRunner(
        data={
            "root": str(root),
            "patterns": "*T1w.nii*",
            "levels": {"dataset": "PT*"},
        }
    )
    assert runner._input_files == [keep]


def test_plan_outputs_skips_when_required_output_exists(tmp_path: Path):
    p = _touch(tmp_path / "scan.nii.gz")
    _touch(tmp_path / "scan_synthseg.nii.gz")  # pretend output exists
    runner = _DummyRunner(data=p, overwrite=False)
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["__inputs__"] == []
    assert plans["seg"] == []


def test_plan_outputs_does_not_skip_when_only_optional_exists(tmp_path: Path):
    p = _touch(tmp_path / "scan.nii.gz")
    _touch(tmp_path / "scan_post.nii.gz")  # optional output
    runner = _DummyRunner(data=p, overwrite=False)
    plans = runner._plan_outputs(
        [
            OutputSpec("seg", "synthseg", save=True, required=True),
            OutputSpec("post", "post", save=False, required=False),
        ]
    )
    assert plans["__inputs__"] == [str(p)]


def test_ensure_output_dirs_creates_parents(tmp_path: Path):
    files = [_touch(tmp_path / "in" / f"{name}.nii.gz") for name in ("a", "b")]
    out_dir = tmp_path / "out"
    runner = _DummyRunner(data=files, output_dir=out_dir)
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    runner._ensure_output_dirs(plans)
    assert out_dir.is_dir()
    for seg in plans["seg"]:
        assert Path(seg).parent.is_dir()


def test_slice_inputs_before_planning(tmp_path: Path, capsys):
    files = [_touch(tmp_path / f"{name}.nii.gz") for name in ("a", "b", "c", "d")]
    runner = _DummyRunner(data=files, output_dir=tmp_path / "out")
    runner._slice_inputs(num_parts=4, part_idx=[0, 2])

    assert runner._input_files == [files[0].resolve(), files[2].resolve()]
    out = capsys.readouterr().out
    assert "Selecting part(s) of 4 discovered input(s)" in out
    assert "Kept 2 of 4 input(s) after part slicing" in out

    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["__inputs__"] == [str(files[0].resolve()), str(files[2].resolve())]
    assert [Path(p).name for p in plans["seg"]] == [
        "a_synthseg.nii.gz",
        "c_synthseg.nii.gz",
    ]


def test_slice_inputs_noop(tmp_path: Path, capsys):
    p = _touch(tmp_path / "scan.nii.gz")
    runner = _DummyRunner(data=p)
    runner._slice_inputs(num_parts=1, part_idx=0)
    assert runner._input_files == [p.resolve()]
    out = capsys.readouterr().out
    assert "Part slicing is a no-op" in out
