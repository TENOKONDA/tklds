from __future__ import annotations

import numpy as np
from typing import Sequence
from scipy.special import ndtri, ndtr
from scipy.stats import norm
from tklds.interface.generators import generate_lds_rvs
from tklds.constant import SequenceNum

def black_scholes_european_price(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float, cp: int = 1) -> float:
    if cp not in (-1, 1):
        raise ValueError("cp must be +1 for a call or -1 for a put")
    vol_sqrt_t = volatility * np.sqrt(expiry)
    d1 = ((np.log(spot / strike)
           + (rate + 0.5 * volatility ** 2) * expiry) / vol_sqrt_t)
    d2 = d1 - vol_sqrt_t

    return cp * (spot * norm.cdf(cp * d1)
                 - strike * np.exp(-rate * expiry) * norm.cdf(cp * d2))


def black_scholes_european_delta(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float, cp: int = 1) -> float:
    if cp not in (-1, 1):
        raise ValueError("cp must be +1 for a call or -1 for a put")
    vol_sqrt_t = volatility * np.sqrt(expiry)
    d1 = ((np.log(spot / strike)
           + (rate + 0.5 * volatility ** 2) * expiry) / vol_sqrt_t)
    return norm.cdf(d1) - (cp - 1) / 2


def black_scholes_european_vega(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float, cp: int = 1) -> float:
    if cp not in (-1, 1):
        raise ValueError("cp must be +1 for a call or -1 for a put")
    vol_sqrt_t = volatility * np.sqrt(expiry)
    d1 = ((np.log(spot / strike)
           + (rate + 0.5 * volatility ** 2) * expiry) / vol_sqrt_t)
    return spot * norm.pdf(d1) * np.sqrt(expiry)


def black_scholes_european_gamma(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float, cp: int = 1) -> float:
    vol_sqrt_t = volatility * np.sqrt(expiry)
    d1 = ((np.log(spot / strike)
           + (rate + 0.5 * volatility ** 2) * expiry) / vol_sqrt_t)
    return (norm.pdf(d1)) / (spot * vol_sqrt_t)


def sobol_normal_matrix(
        num_paths: int, num_dimensions: int,
        sequence: SequenceNum, skip: int = 0 ) -> np.ndarray:
    u = generate_lds_rvs(sequence, num_paths, num_dimensions, skip)
    u = np.asarray(u, dtype=float)
    lo = np.nextafter(0.0, 1.0)
    hi = np.nextafter(1.0, 0.0)
    np.clip(u, lo, hi, out=u)
    return ndtri(u)

def sobol_european_option_prices(
        spot: float, strike: float, rate: float,
        volatility: float, dimension_grid: np.ndarray,
        expiry: float, z: np.ndarray, cp: int = 1,) -> np.ndarray:
    if cp not in (-1, 1):
        raise ValueError("cp must be +1 for a call or -1 for a put")
    dimension_grid = np.asarray(dimension_grid, dtype=int)
    z = np.cumsum(z, axis=1)
    log_spot = np.log(spot)
    drift = (rate - 0.5 * volatility ** 2) * expiry
    discount = np.exp(-rate * expiry)
    scales = volatility * np.sqrt(expiry / dimension_grid.astype(float))
    prices = np.empty(dimension_grid.size, dtype=float)
    for i, (d, scale) in enumerate(zip(dimension_grid, scales)):
        log_terminal = log_spot + drift + scale * z[:, d - 1]
        terminal = np.exp(log_terminal)
        prices[i] = discount * np.maximum(cp * (terminal - strike), 0.0).mean()
    return prices


def sobol_pathwise_european_deltas(
        spot: float, strike: float, rate: float,
        volatility: float, dimension_grid: np.ndarray,
        expiry: float, z:np.ndarray, cp: int = 1,) -> np.ndarray:
    if cp not in (-1, 1):
        raise ValueError("cp must be +1 for a call or -1 for a put")
    dimension_grid = np.asarray(dimension_grid, dtype=int)
    z = np.cumsum(z, axis=1)
    log_spot = np.log(spot)
    drift = (rate - 0.5 * volatility ** 2) * expiry
    discount = np.exp(-rate * expiry)
    scales = volatility * np.sqrt(expiry / dimension_grid.astype(float))
    deltas = np.empty(dimension_grid.size, dtype=float)
    for i, (d, scale) in enumerate(zip(dimension_grid, scales)):
        log_terminal = log_spot + drift + scale * z[:, d - 1]
        terminal = np.exp(log_terminal)
        deltas[i] = discount * np.where(terminal > strike,
                                        terminal / spot, 0.).mean()
    return deltas


def sobol_pathwise_european_vegas(
        spot: float, strike: float, rate: float,
        volatility: float, dimension_grid: np.ndarray,
        expiry: float, z: np.ndarray, cp: int = 1,) -> np.ndarray:
    if cp not in (-1, 1):
        raise ValueError("cp must be +1 for a call or -1 for a put")
    dimension_grid = np.asarray(dimension_grid, dtype=int)
    z = np.cumsum(z, axis=1)
    log_spot = np.log(spot)
    drift = (rate - 0.5 * volatility ** 2) * expiry
    discount = np.exp(-rate * expiry)
    scales = volatility * np.sqrt(expiry / dimension_grid.astype(float))
    vegas = np.empty(dimension_grid.size, dtype=float)
    for i, (d, scale) in enumerate(zip(dimension_grid, scales)):
        log_terminal = log_spot + drift + scale * z[:, d - 1]
        terminal = np.exp(log_terminal)
        vegas[i] = discount * np.where(terminal > strike,
                                       terminal * (
                                               -volatility * expiry +
                                               scale * z[:, d - 1] / volatility),
                                       0.).mean()
    return vegas


def sobol_finite_difference_european_gamma(
        spot: float, strike: float, rate: float,
        volatility: float, dimension_grid: np.ndarray,
        expiry: float, z: np.ndarray, cp: int = 1, bump: float = 0.1) -> np.ndarray:
    if cp not in (-1, 1):
        raise ValueError("cp must be +1 for a call or -1 for a put")
    dimension_grid = np.asarray(dimension_grid, dtype=int)
    z = np.cumsum(z, axis=1)
    log_spot = np.log(spot)
    log_spot_minus = np.log(spot - bump)
    log_spot_plus = np.log(spot + bump)
    drift = (rate - 0.5 * volatility ** 2) * expiry
    discount = np.exp(-rate * expiry)
    scales = volatility * np.sqrt(expiry / dimension_grid.astype(float))
    gammas = np.empty(dimension_grid.size, dtype=float)
    for i, (d, scale) in enumerate(zip(dimension_grid, scales)):
        log_terminal = log_spot + drift + scale * z[:, d - 1]
        log_terminal_m = log_spot_minus + drift + scale * z[:, d - 1]
        log_terminal_p = log_spot_plus + drift + scale * z[:, d - 1]
        terminal = np.exp(log_terminal)
        terminal_m = np.exp(log_terminal_m)
        terminal_p = np.exp(log_terminal_p)
        _prices = discount * np.maximum(cp * (terminal - strike), 0.0).mean()
        _prices_m = discount * np.maximum(cp * (terminal_m - strike), 0.0).mean()
        _prices_p = discount * np.maximum(cp * (terminal_p - strike), 0.0).mean()
        gammas[i] = (_prices_p - 2 * _prices + _prices_m) / (bump * bump)
    return gammas

def run_sobol_price_comparison(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float,
        z_a: np.ndarray, z_b: np.ndarray,
        dimension_grid_a: np.ndarray,
        dimension_grid_b: np.ndarray,
        cp: int = 1,):
    reference = black_scholes_european_price(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, expiry=expiry, cp=cp)
    prices_a = sobol_european_option_prices(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_a,
        dimension_grid=dimension_grid_a, expiry=expiry,
        cp=cp)
    prices_b = sobol_european_option_prices(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_b,
        dimension_grid=dimension_grid_b, expiry=expiry,
        cp=cp)

    return reference, prices_a, prices_b

def run_sobol_delta_comparison(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float,
        z_a: np.ndarray, z_b: np.ndarray,
        dimension_grid_a: np.ndarray,
        dimension_grid_b: np.ndarray,
        cp: int = 1):
    reference = black_scholes_european_delta(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, expiry=expiry, cp=cp)
    deltas_a = sobol_pathwise_european_deltas(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_a,
        dimension_grid=dimension_grid_a, expiry=expiry,
        cp=cp)
    deltas_b = sobol_pathwise_european_deltas(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_b,
        dimension_grid=dimension_grid_b, expiry=expiry,
        cp=cp)

    return reference, deltas_a, deltas_b


def run_sobol_vega_comparison(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float,
        z_a: np.ndarray, z_b: np.ndarray,
        dimension_grid_a: np.ndarray,
        dimension_grid_b: np.ndarray,
        cp: int = 1):
    reference = black_scholes_european_vega(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, expiry=expiry, cp=cp)
    vegas_a = sobol_pathwise_european_vegas(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_a,
        dimension_grid=dimension_grid_a, expiry=expiry,
        cp=cp)
    vegas_b = sobol_pathwise_european_vegas(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_b,
        dimension_grid=dimension_grid_b, expiry=expiry,
        cp=cp)

    return reference, vegas_a, vegas_b

def run_sobol_gamma_comparison(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float,
        z_a: np.ndarray, z_b: np.ndarray,
        dimension_grid_a: np.ndarray,
        dimension_grid_b: np.ndarray,
        cp: int = 1, bump: float = 0.1,):
    reference = black_scholes_european_gamma(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, expiry=expiry, cp=cp)
    gammas_a = sobol_finite_difference_european_gamma(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_a,
        dimension_grid=dimension_grid_a, expiry=expiry,
        cp=cp, bump=bump)
    gammas_b = sobol_finite_difference_european_gamma(
        spot=spot, strike=strike, rate=rate,
        volatility=volatility, z=z_b,
        dimension_grid=dimension_grid_b, expiry=expiry,
        cp=cp, bump=bump)

    return reference, gammas_a, gammas_b

def geometric_asian_log_mean_variance(
        spot: float, rate: float, volatility: float,
        expiry: float, d: int | Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dims = np.asarray(d, dtype=float)
    if np.any(dims <= 0.0):
        raise ValueError("All dimensions must be positive.")
    mu = rate - 0.5 * volatility * volatility
    log_mean = np.log(spot) + mu * expiry * (dims + 1.0) / (2.0 * dims)
    log_var = volatility * volatility * expiry * (dims + 1.0) * (2.0 * dims + 1.0) / (6.0 * dims * dims)
    return log_mean, log_var

def _lognormal_option_price_from_log_mean_variance(
        log_mean: np.ndarray | float, log_var: np.ndarray | float,
        strike: float, rate: float, expiry: float,
        cp: int = 1) -> np.ndarray | float:
    log_mean = np.asarray(log_mean, dtype=float)
    log_var = np.asarray(log_var, dtype=float)
    log_std = np.sqrt(log_var)
    d2 = (log_mean - np.log(strike)) / log_std
    d1 = d2 + log_std
    discount = np.exp(-rate * expiry)
    return cp * discount * (
        np.exp(log_mean + 0.5 * log_var) * ndtr(cp * d1) - strike * ndtr(cp * d2)
    )

def geometric_asian_option_price_closed_form(
        spot: float, strike: float, rate: float, volatility: float,
        expiry: float, d: int | Sequence[int] | np.ndarray,
        cp: int = 1,
) -> np.ndarray | float:
    log_mean, log_var = geometric_asian_log_mean_variance(
        spot=spot,
        rate=rate,
        volatility=volatility,
        expiry=expiry,
        d=d,
    )
    return _lognormal_option_price_from_log_mean_variance(
        log_mean=log_mean,
        log_var=log_var,
        strike=strike,
        rate=rate,
        expiry=expiry,
        cp=cp,
    )

def _geometric_asian_price_sweep_from_normals(
        spot: float, strike: float, rate: float, volatility: float,
        expiry: float, z: np.ndarray, dimension_grid: Sequence[int] | np.ndarray,
        cp: int = 1) -> np.ndarray:
    dimension_grid = np.asarray(dimension_grid, dtype=int)
    z = np.asarray(z, dtype=float)

    log_spot = np.log(spot)
    discount = np.exp(-rate * expiry)
    drift = rate - 0.5 * volatility * volatility
    sigma_sqrt_t = volatility * np.sqrt(expiry)

    cumulative = np.zeros(z.shape[0], dtype=float)
    cumulative_weighted = np.zeros(z.shape[0], dtype=float)
    prices = np.empty(dimension_grid.size, dtype=float)

    out_idx = 0
    for d in range(1, int(dimension_grid[-1]) + 1):
        column = z[:, d - 1]
        cumulative += column
        cumulative_weighted += d * column

        if d != dimension_grid[out_idx]:
            continue

        weighted_prefix = (d + 1.0) * cumulative - cumulative_weighted
        normalized_bridge = weighted_prefix / (d * np.sqrt(d))
        log_geometric_average = (
            log_spot
            + drift * expiry * (d + 1.0) / (2.0 * d)
            + sigma_sqrt_t * normalized_bridge
        )
        geometric_average = np.exp(log_geometric_average)
        prices[out_idx] = discount * np.maximum(cp * (geometric_average - strike), 0.0).mean()

        out_idx += 1
        if out_idx == dimension_grid.size:
            break

    return prices


def run_sobol_geometric_asian_price_comparison(
        spot: float, strike: float, rate: float, volatility: float,
        expiry: float, z_a: np.ndarray, z_b: np.ndarray,
        dimension_grid_a: Sequence[int] | np.ndarray, dimension_grid_b: Sequence[int] | np.ndarray,
        cp: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_a = np.asarray(dimension_grid_a, dtype=int)
    grid_b = np.asarray(dimension_grid_b, dtype=int)

    reference = np.asarray(
        geometric_asian_option_price_closed_form(
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=volatility,
            expiry=expiry,
            d=grid_a,
            cp=cp,
        ),
        dtype=float,
    )
    prices_a = _geometric_asian_price_sweep_from_normals(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        expiry=expiry,
        z=z_a,
        dimension_grid=grid_a,
        cp=cp,
    )
    prices_b = _geometric_asian_price_sweep_from_normals(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        expiry=expiry,
        z=z_b,
        dimension_grid=grid_b,
        cp=cp,
    )
    return reference, prices_a, prices_b

def historical_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    arr = np.asarray(losses, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("losses must contain at least one observation.")

    index = int(np.ceil(alpha * arr.size)) - 1
    index = max(0, min(index, arr.size - 1))
    partitioned = np.partition(arr, index)
    var = float(partitioned[index])
    es = float(partitioned[index:].mean())
    return var, es


def short_geometric_asian_var_es_closed_form(
        spot: float, strike: float, rate: float,
        volatility: float, expiry: float,
        d: int | Sequence[int] | np.ndarray,
        alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    dims = np.asarray(d, dtype=float)
    log_mean, log_var = geometric_asian_log_mean_variance(
        spot=spot,
        rate=rate,
        volatility=volatility,
        expiry=expiry,
        d=dims,
    )
    log_std = np.sqrt(log_var)
    discount = np.exp(-rate * expiry)
    premium = np.asarray(
        geometric_asian_option_price_closed_form(
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=volatility,
            expiry=expiry,
            d=dims,
            cp=1,
        ),
        dtype=float,
    )

    z_alpha = ndtri(alpha)
    mass_at_negative_premium = ndtr((np.log(strike) - log_mean) / log_std)

    quantile_geometric_average = np.exp(log_mean + log_std * z_alpha)
    var_positive_tail = discount * (quantile_geometric_average - strike) - premium
    es_positive_tail = (discount * ( np.exp(log_mean + 0.5 * log_var) * ndtr(log_std - z_alpha)
                                     / (1.0 - alpha) - strike) - premium)

    var = np.where(alpha <= mass_at_negative_premium, -premium, var_positive_tail)
    es = np.where(alpha <= mass_at_negative_premium, premium * alpha / (1.0 - alpha), es_positive_tail)
    return np.asarray(var, dtype=float), np.asarray(es, dtype=float)


def _short_geometric_asian_var_es_sweep_from_normals(
        spot: float, strike: float, rate: float, volatility: float,
        expiry: float, z: np.ndarray, dimension_grid: Sequence[int] | np.ndarray,
        alpha: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(dimension_grid, dtype=int)
    z = np.asarray(z, dtype=float)

    log_spot = np.log(spot)
    discount = np.exp(-rate * expiry)
    drift = rate - 0.5 * volatility * volatility
    sigma_sqrt_t = volatility * np.sqrt(expiry)

    cumulative = np.zeros(z.shape[0], dtype=float)
    cumulative_weighted = np.zeros(z.shape[0], dtype=float)
    var_values = np.empty(grid.size, dtype=float)
    es_values = np.empty(grid.size, dtype=float)

    out_idx = 0
    for d in range(1, int(grid[-1]) + 1):
        column = z[:, d - 1]
        cumulative += column
        cumulative_weighted += d * column

        if d != grid[out_idx]:
            continue

        weighted_prefix = (d + 1.0) * cumulative - cumulative_weighted
        normalized_bridge = weighted_prefix / (d * np.sqrt(d))
        log_geometric_average = (log_spot + drift * expiry * (d + 1.0) / (2.0 * d)
                                 + sigma_sqrt_t * normalized_bridge)
        geometric_average = np.exp(log_geometric_average)
        discounted_payoff = discount * np.maximum(geometric_average - strike, 0.0)
        premium = float(
            geometric_asian_option_price_closed_form(
                spot=spot,
                strike=strike,
                rate=rate,
                volatility=volatility,
                expiry=expiry,
                d=d,
                cp=1,
            )
        )
        losses = discounted_payoff - premium
        var_values[out_idx], es_values[out_idx] = historical_var_es(losses, alpha)

        out_idx += 1
        if out_idx == grid.size:
            break

    return var_values, es_values


def run_sobol_short_geometric_asian_var_es_comparison(
        spot: float, strike: float, rate: float, volatility: float,
        expiry: float, alpha: float, z_a: np.ndarray, z_b: np.ndarray,
        dimension_grid_a: Sequence[int] | np.ndarray, dimension_grid_b: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_a = np.asarray(dimension_grid_a, dtype=int)
    grid_b = np.asarray(dimension_grid_b, dtype=int)

    reference_var, reference_es = short_geometric_asian_var_es_closed_form(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        expiry=expiry,
        d=grid_a,
        alpha=alpha,
    )
    var_a, es_a = _short_geometric_asian_var_es_sweep_from_normals(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        expiry=expiry,
        z=z_a,
        dimension_grid=grid_a,
        alpha=alpha,
    )
    var_b, es_b = _short_geometric_asian_var_es_sweep_from_normals(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        expiry=expiry,
        z=z_b,
        dimension_grid=grid_b,
        alpha=alpha,
    )
    return reference_var, reference_es, var_a, var_b, es_a, es_b