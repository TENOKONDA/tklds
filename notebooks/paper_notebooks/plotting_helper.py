from typing import Optional, Sequence, Union, Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _coerce_y_to_mean_sd_reps(
    y: Union[np.ndarray, Sequence[float]],
    x: np.ndarray,
    label: str,
    replicate_axis: Union[int, Literal["auto"]] = "auto",
):
    """
    Returns y_mean, y_sd, y_reps.

    If y is 1D:
        y_mean has shape (n_x,), y_sd and y_reps are None.

    If y is 2D:
        y_reps is returned with shape (n_replicates, n_x).
    """
    y_arr = np.asarray(y, dtype=float)

    if y_arr.ndim == 1:
        y_mean = y_arr.reshape(-1)
        if y_mean.shape != x.shape:
            raise ValueError(
                f"x and y must have the same shape for series '{label}'. "
                f"Got x.shape={x.shape}, y.shape={y_mean.shape}."
            )
        return y_mean, None, None

    if y_arr.ndim != 2:
        raise ValueError(
            f"y for series '{label}' must be either 1D or 2D. "
            f"Got y.ndim={y_arr.ndim}."
        )

    if replicate_axis == "auto":
        if y_arr.shape[1] == x.size:
            y_reps = y_arr
        elif y_arr.shape[0] == x.size:
            y_reps = y_arr.T
        else:
            raise ValueError(
                f"Could not infer replicate axis for series '{label}'. "
                f"For 2D y, one axis must have length len(x)={x.size}. "
                f"Got y.shape={y_arr.shape}."
            )
    elif replicate_axis == 0:
        # rows are randomized replicates; columns correspond to x
        if y_arr.shape[1] != x.size:
            raise ValueError(
                f"When replicate_axis=0, y.shape[1] must equal len(x) for "
                f"series '{label}'. Got y.shape={y_arr.shape}, len(x)={x.size}."
            )
        y_reps = y_arr
    elif replicate_axis == 1:
        # columns are randomized replicates; rows correspond to x
        if y_arr.shape[0] != x.size:
            raise ValueError(
                f"When replicate_axis=1, y.shape[0] must equal len(x) for "
                f"series '{label}'. Got y.shape={y_arr.shape}, len(x)={x.size}."
            )
        y_reps = y_arr.T
    else:
        raise ValueError("replicate_axis must be 'auto', 0, or 1.")

    if y_reps.shape[0] < 1:
        raise ValueError(f"2D y for series '{label}' contains no replicates.")

    ddof = 1 if y_reps.shape[0] > 1 else 0
    y_mean = np.mean(y_reps, axis=0)
    y_sd = np.std(y_reps, axis=0, ddof=ddof)

    return y_mean, y_sd, y_reps


def _update_min_positive(current, *arrays):
    for arr in arrays:
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            candidate = float(vals.min())
            current = candidate if current is None else min(current, candidate)
    return current


def plot_with_error_panel(
    x_data: Sequence[np.ndarray],
    y_data: Sequence[np.ndarray],
    y_axis_label_top: str,
    y_axis_label_bottom: str,
    x_axis_label: str,
    y_data_label: Sequence[str],
    y_data_colors: Sequence[str],
    reference_x_data: Optional[np.ndarray] = None,
    reference_y_data: Union[float, np.ndarray] = 0.0,
    reference_y_data_label: str = "reference",
    reference_y_data_color: str = "black",
    x_axis_log: bool = True,
    y_axis_log: bool = True,
    title: str = "",
    save: bool = False,
    file_name: Optional[str] = None,
    file_type: Optional[str] = None,
    plot_params_dict: Optional[dict] = None,
    replicate_axis: Union[int, Literal["auto"]] = "auto",
    y_band_scale: float = 1.0,
    y_band_stat: Literal["sd", "se"] = "sd",
    band_alpha: float = 0.18,
    band_in_legend: bool = False,
    show_replicate_error_band: bool = True,
    error_band_quantiles: Tuple[float, float] = (
        0.15865525393145707,
        0.8413447460685429,
    ),
):
    """
    Plot one or more deterministic or randomized series with an error panel.

    Backward-compatible behavior
    ----------------------------
    If an element of y_data is 1D with shape (n_x,), it is plotted as a
    deterministic series.

    Randomized-replicate behavior
    -----------------------------
    If an element of y_data is 2D, it is treated as a matrix of randomized
    replicate curves. The top panel plots the replicate mean, with a filled band
    equal to mean +/- y_band_scale * replicate standard deviation by default.

    Expected 2D shape:
        replicate_axis=0:
            y.shape == (n_replicates, n_x)

        replicate_axis=1:
            y.shape == (n_x, n_replicates)

        replicate_axis="auto":
            infer the axis matching len(x)

    The bottom panel plots abs(mean - reference). If show_replicate_error_band
    is True, it also fills the central quantile band of replicate absolute errors.
    """
    if not (
        len(x_data) == len(y_data) == len(y_data_label) == len(y_data_colors)
    ):
        raise ValueError(
            "x_data, y_data, y_data_label, and y_data_colors must have the same length."
        )

    if y_band_scale < 0:
        raise ValueError("y_band_scale must be non-negative.")

    if y_band_stat not in {"sd", "se"}:
        raise ValueError("y_band_stat must be either 'sd' or 'se'.")

    q_low, q_high = error_band_quantiles
    if not (0.0 <= q_low <= q_high <= 1.0):
        raise ValueError("error_band_quantiles must satisfy 0 <= low <= high <= 1.")

    default_rc = {
        "figure.figsize": (10, 7.5),
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    }

    rc = default_rc.copy()
    if plot_params_dict:
        rc.update(plot_params_dict)

    with plt.rc_context(rc):
        fig, (ax, ax_errors) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )

        ref_arr = np.asarray(reference_y_data, dtype=float)
        scalar_ref = (ref_arr.ndim == 0) or (ref_arr.size == 1)

        if scalar_ref:
            ref_value = float(ref_arr.reshape(-1)[0])
            ax.axhline(
                ref_value,
                linestyle="--",
                color=reference_y_data_color,
                label=reference_y_data_label,
                linewidth=1.25,
            )
            ref_x = None
            ref_y = None
        else:
            if reference_x_data is None:
                raise ValueError(
                    "reference_x_data is required when reference_y_data is an array."
                )

            ref_x = np.asarray(reference_x_data, dtype=float).reshape(-1)
            ref_y = ref_arr.reshape(-1)

            if ref_x.shape != ref_y.shape:
                raise ValueError(
                    "reference_x_data and reference_y_data must have the same shape."
                )

            order = np.argsort(ref_x)
            ref_x = ref_x[order]
            ref_y = ref_y[order]

            ax.plot(
                ref_x,
                ref_y,
                linestyle="--",
                color=reference_y_data_color,
                label=reference_y_data_label,
                linewidth=1.25,
            )

        series_records = []
        min_positive_error = None

        for x_raw, y_raw, label, color in zip(
            x_data, y_data, y_data_label, y_data_colors
        ):
            x = np.asarray(x_raw, dtype=float).reshape(-1)

            if x.size == 0:
                raise ValueError(f"x for series '{label}' is empty.")

            y_mean, y_sd, y_reps = _coerce_y_to_mean_sd_reps(
                y_raw,
                x,
                label,
                replicate_axis=replicate_axis,
            )

            # Sort each series so fill_between behaves cleanly.
            order = np.argsort(x)
            x = x[order]
            y_mean = y_mean[order]

            if y_sd is not None:
                y_sd = y_sd[order]

            if y_reps is not None:
                y_reps = y_reps[:, order]

            if scalar_ref:
                ref_vals = np.full_like(x, ref_value, dtype=float)
            else:
                ref_vals = np.interp(x, ref_x, ref_y)

            y_error = np.abs(y_mean - ref_vals)

            error_band = None
            if y_reps is not None and show_replicate_error_band:
                replicate_abs_errors = np.abs(y_reps - ref_vals[None, :])
                error_band = tuple(
                    np.quantile(
                        replicate_abs_errors,
                        [q_low, q_high],
                        axis=0,
                    )
                )
                min_positive_error = _update_min_positive(
                    min_positive_error,
                    y_error,
                    error_band[0],
                    error_band[1],
                )
            else:
                min_positive_error = _update_min_positive(
                    min_positive_error,
                    y_error,
                )

            series_records.append(
                {
                    "x": x,
                    "y_mean": y_mean,
                    "y_sd": y_sd,
                    "y_reps": y_reps,
                    "y_error": y_error,
                    "error_band": error_band,
                    "label": label,
                    "color": color,
                }
            )

        # Top panel: mean line plus optional replicate band.
        for rec in series_records:
            x = rec["x"]
            y_mean = rec["y_mean"]
            y_sd = rec["y_sd"]
            y_reps = rec["y_reps"]
            label = rec["label"]
            color = rec["color"]

            if y_sd is not None:
                band_width = y_sd.copy()

                if y_band_stat == "se":
                    band_width = band_width / np.sqrt(y_reps.shape[0])

                lower = y_mean - y_band_scale * band_width
                upper = y_mean + y_band_scale * band_width

                band_label = (
                    f"{label} ± {y_band_scale:g} {y_band_stat}"
                    if band_in_legend
                    else None
                )

                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=band_alpha,
                    linewidth=0,
                    label=band_label,
                )

            ax.plot(
                x,
                y_mean,
                linewidth=1.75,
                color=color,
                label=label,
            )

        use_log_error_axis = y_axis_log and (min_positive_error is not None)
        floor = 0.5 * min_positive_error if use_log_error_axis else None

        # Bottom panel: error of the mean plus optional replicate absolute-error band.
        for rec in series_records:
            x = rec["x"]
            y_error = rec["y_error"]
            color = rec["color"]
            error_band = rec["error_band"]

            if error_band is not None:
                lo, hi = error_band

                if use_log_error_axis:
                    lo = np.where(lo > 0, lo, floor)
                    hi = np.where(hi > 0, hi, floor)

                ax_errors.fill_between(
                    x,
                    lo,
                    hi,
                    color=color,
                    alpha=band_alpha,
                    linewidth=0,
                )

            y_plot = (
                np.where(y_error > 0, y_error, floor)
                if use_log_error_axis
                else y_error
            )

            ax_errors.plot(
                x,
                y_plot,
                linewidth=1.5,
                linestyle="--",
                color=color,
            )

        ax.set_title(title, pad=15)
        ax.set_ylabel(y_axis_label_top)
        ax.grid(True, which="both")
        ax.legend(loc="best", fontsize=12, frameon=True)

        ax_errors.set_ylabel(y_axis_label_bottom)
        ax_errors.set_xlabel(x_axis_label)
        ax_errors.grid(True, which="both")

        if x_axis_log:
            ax.set_xscale("log", base=2)
            ax_errors.set_xscale("log", base=2)

        if use_log_error_axis:
            ax_errors.set_yscale("log", base=10)

        if save:
            ext = (file_type or "png").lstrip(".")
            out = file_name or "plot"

            if not out.lower().endswith(f".{ext.lower()}"):
                out = f"{out}.{ext}"

            fig.savefig(out, format=ext, bbox_inches="tight")

        plt.show()
        return fig, (ax, ax_errors)