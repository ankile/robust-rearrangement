from src.dataset.normalizer import LinearNormalizer, GaussianNormalizer, Normalizer


def get_normalizer(normalizer_type, control_mode) -> Normalizer:
    assert normalizer_type in [
        "min_max",
        "mean_std",
    ], f"Normalizer type {normalizer_type} not recognized."

    if normalizer_type == "min_max":
        return LinearNormalizer(control_mode=control_mode)

    if normalizer_type == "mean_std":
        return GaussianNormalizer(control_mode=control_mode)

    raise ValueError(f"Normalizer type {normalizer_type} not recognized.")
