"""HD-BET-specific helpers (citation banner)."""

from __future__ import annotations

from typing import Callable


def print_hd_bet_citation(emit: Callable[[str], None] = print) -> None:
    """Emit the HD-BET citation block line by line.

    Args:
        emit: Sink for each citation line. Defaults to :func:`print`. Pass a
            logger-bound callable (e.g., ``logger.info``) to route the citation
            through the wrapper logger.
    """
    hd_bet_citation = (
        "HD-BET (MIC-DKFZ): If you use HD-BET, please cite:\n"
        "  Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A,\n"
        "  Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P.\n"
        "  Automated brain extraction of multi-sequence MRI using artificial neural\n"
        "  networks. Hum Brain Mapp. 2019; 1-13. https://doi.org/10.1002/hbm.24750\n"
        "  Isensee F, Jaeger PF, Kohl SA, Petersen J, Maier-Hein KH. nnU-Net: a\n"
        "  self-configuring method for deep learning-based biomedical image\n"
        "  segmentation. Nature Methods 2021;18(2):203-211."
    )
    emit("########################")
    for line in hd_bet_citation.splitlines():
        emit(line)
    emit("########################")
