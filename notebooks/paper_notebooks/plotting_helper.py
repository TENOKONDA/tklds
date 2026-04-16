from typing import Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt

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
):
    if not (
        len(x_data) == len(y_data) == len(y_data_label) == len(y_data_colors)
    ):
        raise ValueError(
            "x_data, y_data, y_data_label, and y_data_colors must have the same length."
        )

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

    with plt.rc_context(plot_params_dict or default_rc):
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

        error_series = []
        min_positive_error = None

        for x, y, label, color in zip(x_data, y_data, y_data_label, y_data_colors):
            x = np.asarray(x, dtype=float).reshape(-1)
            y = np.asarray(y, dtype=float).reshape(-1)

            if x.shape != y.shape:
                raise ValueError(f"x and y must have the same shape for series '{label}'.")

            if scalar_ref:
                y_error = np.abs(y - ref_value)
            else:
                # Interpolate reference curve onto this series' x-grid
                y_ref_interp = np.interp(x, ref_x, ref_y)
                y_error = np.abs(y - y_ref_interp)

            ax.plot(
                x,
                y,
                linewidth=1.75,
                color=color,
                label=label,
            )

            error_series.append((x, y_error, color))

            pos = y_error[y_error > 0]
            if pos.size:
                cur_min = pos.min()
                min_positive_error = (
                    cur_min if min_positive_error is None
                    else min(min_positive_error, cur_min)
                )

        use_log_error_axis = y_axis_log and (min_positive_error is not None)
        floor = 0.5 * min_positive_error if use_log_error_axis else None

        for x, y_error, color in error_series:
            y_plot = np.where(y_error > 0, y_error, floor) if use_log_error_axis else y_error
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