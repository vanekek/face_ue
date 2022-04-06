from .distance_uncertainty_funcs import (
    l2_normalize,
    pair_euc_score,
    pair_cosine_score,
    pair_centered_cosine_score,
    pair_MLS_score,
    pair_uncertainty_sum,
    pair_uncertainty_squared_sum,
    pair_uncertainty_mul,
    pair_uncertainty_harmonic_sum,
    pair_uncertainty_harmonic_mul,
    pair_uncertainty_concatenated_harmonic,
    pair_uncertainty_squared_harmonic,
    pair_uncertainty_cosine_analytic,
    pair_scale_mul_cosine_score,
    pair_scale_harmonic_cosine_score,
    pair_sqrt_scale_mul_cosine_score,
    pair_sqrt_scale_harmonic_cosine_score,
    pair_scale_mul_centered_cosine_score,
    pair_scale_harmonic_centered_cosine_score,
    pair_sqrt_scale_mul_centered_cosine_score,
    pair_sqrt_scale_harmonic_centered_cosine_score,
    pair_uncertainty_min
)

name_to_distance_func = {
    "euc": pair_euc_score,
    "cosine": pair_cosine_score,
    "centered-cosine": pair_centered_cosine_score,
    "MLS": pair_MLS_score,
    "scale-mul-cosine": pair_scale_mul_cosine_score,
    "scale-harmonic-cosine": pair_scale_harmonic_cosine_score,
    "scale-sqrt-mul-cosine": pair_sqrt_scale_mul_cosine_score,
    "scale-sqrt-harmonic-cosine": pair_sqrt_scale_harmonic_cosine_score,
    "scale-mul-centered-cosine": pair_scale_mul_centered_cosine_score,
    "scale-harmonic-centered-cosine": pair_scale_harmonic_centered_cosine_score,
    "scale-sqrt-mul-centered-cosine": pair_sqrt_scale_mul_centered_cosine_score,
    "scale-sqrt-harmonic-centered-cosine": pair_sqrt_scale_harmonic_centered_cosine_score,
}

name_to_uncertainty_func = {
    "mean": pair_uncertainty_sum,
    "squared-sum": pair_uncertainty_squared_sum,
    "mul": pair_uncertainty_mul,
    "harmonic-sum": pair_uncertainty_harmonic_sum,
    "harmonic-mul": pair_uncertainty_harmonic_mul,
    "harmonic-harmonic": pair_uncertainty_concatenated_harmonic,
    "squared-harmonic": pair_uncertainty_squared_harmonic,
    "cosine-analytic": pair_uncertainty_cosine_analytic,
    "min": pair_uncertainty_min
}