"""Selected-configuration scorer used after SEARCH configurations are frozen."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import scripts.search_harmonic_metric_broad_parameters_v0 as core


def selected_scores(
    piece_ids: Sequence[str], cache_root: Path, configs: pd.DataFrame, audit: pd.DataFrame
) -> pd.DataFrame:
    performance = audit.set_index("piece_id").performance_id.to_dict()
    rows = []
    for piece_id in sorted(piece_ids):
        for system in core.SYSTEMS:
            stats = (
                None
                if system == "NO_PEDAL"
                else core.load_stats(cache_root / f"{piece_id}__{system}.npz")
            )
            for config in configs.itertuples(index=False):
                score = 0.0 if stats is None else core.score_stats(
                    stats,
                    float(config.alpha_decay),
                    float(config.eta_PP),
                    int(config.m0),
                    float(config.beta_low),
                    float(config.kappa_dyn),
                )
                rows.append(
                    {
                        "piece_id": piece_id,
                        "performance_id": performance[piece_id],
                        "system": system,
                        "config_id": config.config_id,
                        "H_piece": score,
                    }
                )
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame.H_piece).all():
        raise AssertionError("selected-config score contains non-finite value")
    if not (frame[frame.system == "NO_PEDAL"].H_piece == 0.0).all():
        raise AssertionError("NO_PEDAL is not exactly zero")
    return frame
