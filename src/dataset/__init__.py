from typing import Literal

import src.dataset.normalizer as N


def get_normalizer(
    normalizer_type: Literal["min_max", "mean_std", "none"],
    control_mode: Literal["delta", "pos"],
) -> N.Normalizer:
    assert normalizer_type in [
        "min_max",
        "mean_std",
        "none",
    ], f"Normalizer type {normalizer_type} not recognized."

    if normalizer_type == "min_max":
        return N.LinearNormalizer(control_mode=control_mode)

    if normalizer_type == "mean_std":
        return N.GaussianNormalizer(control_mode=control_mode)

    if normalizer_type == "none":
        return N.NoNormalizer(control_mode=control_mode)
