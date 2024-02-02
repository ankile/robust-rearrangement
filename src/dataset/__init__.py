from src.dataset.normalizer import LinearNormalizer, GaussianNormalizer


def get_normalizer_cls(normalizer_type):
    assert normalizer_type in [
        "min_max",
        "mean_std",
    ], f"Normalizer type {normalizer_type} not recognized."

    if normalizer_type == "min_max":
        return LinearNormalizer

    if normalizer_type == "mean_std":
        return GaussianNormalizer

    raise ValueError(f"Normalizer type {normalizer_type} not recognized.")
