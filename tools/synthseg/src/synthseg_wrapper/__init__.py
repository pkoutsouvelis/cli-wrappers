"""SynthSeg wrapper coupling Photo-SynthSeg (TF 2.15) with nifti-finder discovery."""

from synthseg_wrapper.config import load_config
from synthseg_wrapper.runner import SynthSegRunner

__all__ = ["SynthSegRunner", "load_config"]
__version__ = "0.1.0"
