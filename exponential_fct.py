import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

single_rate = True
fixed_rates = False

def _exp_model(x, params):
    if single_rate:
        A, B, k = params
        return A + B * np.exp(-k * x)
    else:
        A, B, k1, C, k2 = params
        return A + B * np.exp(-k1 * x) + C * np.exp(-k2 * x)

def _exp_model_dual(x, params):
    A, B, k1, C, k2 = params
    return A + B * np.exp(-k1 * x) + C * np.exp(-k2 * x)

def _fit_exp(xs: np.ndarray, ys: np.ndarray, x_len: int) -> Tuple[float, float, float]:
    if single_rate:
        return _fit_single_exp(xs, ys, x_len)
    else:
        return _fit_dual_exp(xs, ys, x_len)

def _fit_dual_exp(xs: np.ndarray, ys: np.ndarray, x_len: int) -> Tuple[float, float, float, float, float]:
    """
    Fit y = A + B * exp(-k1 * t) + C * exp(-k2 * t).
    Constrains amplitudes B and C to be positive to prevent cancellation effects.
    Uses fixed time constants k1=5 and k2=60 (rates 1/5 and 1/60).
    """
    A0 = float(np.median(ys[-x_len:]))
    total_amplitude = max(ys.max() - A0, 1e-3)
    B0 = total_amplitude * 0.5
    C0 = total_amplitude * 0.5

    if fixed_rates:
        # Fixed time constants
        k1_fixed = 1.0 / 5.0
        k2_fixed = 1.0 / 60.0

        def _fixed_k_model(x, A, B, C):
            return A + B * np.exp(-k1_fixed * x) + C * np.exp(-k2_fixed * x)

        try:
            popt, _ = curve_fit(
                _fixed_k_model, xs, ys,
                p0=(A0, B0, C0),
                bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
            A, B, C = map(float, popt)
            return [A, B, k1_fixed, C, k2_fixed]

        except Exception:
            A, B, k = _fit_single_exp(xs, ys, x_len)
            return [A, B, k, 0.0, 0.0]
    else:
        # Dynamic time constants
        k1_0 = 1.0 / 5.0
        k2_0 = 1.0 / 60.0

        def _dynamic_k_model(x, A, B, k1, C, k2):
            return A + B * np.exp(-k1 * x) + C * np.exp(-k2 * x)

        try:
            popt, _ = curve_fit(
                _dynamic_k_model, xs, ys,
                p0=(A0, B0, k1_0, C0, k2_0),
                bounds=([0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
            A, B, k1, C, k2 = map(float, popt)
            return [A, B, k1, C, k2]

        except Exception:
            A, B, k = _fit_single_exp(xs, ys, x_len)
            return [A, B, k, 0.0, 0.0]

def _fit_single_exp(xs: np.ndarray, ys: np.ndarray, x_len: int) -> Tuple[float, float, float]:
    """
    Fit y = A + B * exp(-k t). Tries SciPy; falls back to NumPy-only approach.
    """
    A0 = float(np.median(ys[-x_len:]))          # late-time plateau guess
    B0 = float(max(ys.max() - A0, 1e-3))        # amplitude guess
    k0 = 0.1                                     # rate guess

    def _single_exp(x, a, b, k):
        return a + b * np.exp(-k * x)

    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(
            _single_exp, xs, ys,
            p0=(A0, B0, k0),
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        A, B, k = map(float, popt)
        return [A, B, k]
    except Exception:
        # NumPy fallback: scan A, linearize ln(ys - A) = ln B - k t
        A_candidates = np.linspace(np.percentile(ys, 20), np.percentile(ys, 80), 60)
        best = None
        for A in A_candidates:
            y_shift = ys - A
            mask = y_shift > 1e-8
            if mask.sum() < 5:
                continue
            t_sel = xs[mask]
            ln_sel = np.log(y_shift[mask])
            slope, intercept = np.polyfit(t_sel, ln_sel, 1)  # ln = intercept + slope * t
            B = float(np.exp(intercept))
            k = float(-slope)
            sse = float(np.sum((ys - _single_exp(xs, A, B, k)) ** 2))
            if (best is None) or (sse < best[0]):
                best = (sse, float(A), float(B), float(k))

        if best is None:
            # Degenerate fallback: flat line
            return float(np.mean(ys)), 0.0, 0.0

        _, A, B, k = best
        return [A, B, k]

def _const_model(x, params):
    """Horizontal line y = a."""
    (a,) = params
    return np.full_like(np.asarray(x, dtype=float), a, dtype=float)

def _fit_const(ys: np.ndarray) -> Tuple[float]:
    """Fit a horizontal line y = a. The least-squares estimate is the mean."""
    return [float(np.mean(ys))]

def _line_model(x, params):
    """Line with nonzero slope y = a + b * t."""
    a, b = params
    return a + b * np.asarray(x, dtype=float)

def _fit_line(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """Fit a line y = a + b * t via ordinary least squares."""
    slope, intercept = np.polyfit(xs, ys, 1)  # polyfit returns [slope, intercept]
    return [float(intercept), float(slope)]

# Offset added to prevent division-by-zero errors
POWER_OFFSET = 1.0
def _power_model(x, params):
    """Power function y = a + b * (t + POWER_OFFSET) ** (-c)."""
    a, b, c = params
    T = np.asarray(x, dtype=float) + POWER_OFFSET
    return a + b * np.power(T, -c)

def _fit_power(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y = a + b * (t + POWER_OFFSET) ** (-c) via nonlinear least squares.
    The offset avoids division-by-zero errors when t = 0.
    """
    A0 = float(np.median(ys))                 # plateau guess
    B0 = float(max(ys.max() - A0, 1e-3))      # amplitude guess
    c0 = 0.24                                  # exponent guess (paper value)

    def _power(x, a, b, c):
        T = np.asarray(x, dtype=float) + POWER_OFFSET
        return a + b * np.power(T, -c)

    try:
        popt, _ = curve_fit(
            _power, xs, ys,
            p0=(A0, B0, c0),
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        a, b, c = map(float, popt)
        return [a, b, c]
    except Exception:
        # Degenerate fallback: flat line at the mean
        return [float(np.mean(ys)), 0.0, 0.0]

def bootstrap_ci_for_group(
    data_dict: Dict[str, List[float]],
    t_eval: np.ndarray,
    n_boot: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    x_base = np.array([round(0.00 + (0.6 + 4.8 + 0.6) * i, 1) for i in range(13)], dtype=float)

    # Group data by time point
    time_groups = {}
    for ys in data_dict.values():
        arr = np.asarray(ys, dtype=float)
        for i, y in enumerate(arr):
            if i < len(x_base):
                if not np.isfinite(y):
                    continue
                t = x_base[i]
                if t not in time_groups:
                    time_groups[t] = []
                time_groups[t].append(y)

    # Drop time points that ended up empty after filtering
    time_groups = {t: vals for t, vals in time_groups.items() if len(vals) > 0}

    if not time_groups:
        raise ValueError("No data available for bootstrap.")

    samples = []
    all_params = []
    # do time-point level resampling
    while len(samples) < n_boot:
        xs_boot_list = []
        ys_boot_list = []

        for t, y_vals in time_groups.items():
            if not y_vals:
                continue
            # Resample with replacement from the values at this time point
            resampled_ys = rng.choice(y_vals, size=len(y_vals), replace=True)
            xs_boot_list.append(np.full(len(resampled_ys), t))
            ys_boot_list.append(resampled_ys)

        if not xs_boot_list:
            continue

        xs_boot = np.concatenate(xs_boot_list)
        ys_boot = np.concatenate(ys_boot_list)

        try:
            params = _fit_exp(xs_boot, ys_boot, x_len=len(x_base))
            A = params[0]
            tt5 = np.log(20.0)/params[2] if params[2] != 0 else np.nan
            # Remove extreme outliers
            if tt5 > 300 or A < 0.05 or not np.isfinite(A) or not np.isfinite(tt5):
                continue
            else:
                all_params.append(params)
                yb = _exp_model(t_eval, params)
                samples.append(yb)
        except Exception:
            # occasionally a fit can fail; just skip that replicate
            continue

    # if len(samples) < max(50, n_boot // 10):
    #     raise RuntimeError(f"Too few successful bootstrap fits ({len(samples)}/{n_boot}).")

    if not samples:
        raise RuntimeError("No successful bootstrap fits after filtering non-finite values.")

    S = np.vstack(samples)  # [n_ok, len(t_eval)]
    lo = np.percentile(S, 2.5, axis=0)
    hi = np.percentile(S, 97.5, axis=0)
    return lo, hi, all_params

def plot_eval_and_fit(
    data_eval_avg: Dict[str, List[float]],
    title: str = "Evaluation errors",
    save_path: Optional[str] = None,
    marker_size = 3,
    group_color = None,
    group_marker = None,
    colors = None,
    split = False,
    hide_yaxis = False,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    data_eval_avg : dict
        {person: list of 13 floats}
    save_path : str or None
        If provided, saves the figure to this path. No plt.show().

    Returns
    -------
    dict
        {"A": A, "B": B, "k": k} fitted parameters.
    """
    # x values: x_i = 0.6 + (0.6 + 4.8 + 0.6) * i, rounded to 1 decimal, i=0..12
    # x = np.array([round(0.6 + (0.6 + 4.8 + 0.6) * i, 1) for i in range(13)], dtype=float)

    # # Flatten for fitting
    # xs_fit = np.tile(x, len(data_eval_avg))
    # ys_fit = np.concatenate([np.asarray(v, dtype=float) for v in data_eval_avg.values()])

    x_base = np.array([round(0.00 + (0.6 + 4.8 + 0.6) * i, 1) for i in range(13)], dtype=float)

    # ---- FIX: build xs/ys per participant length (handles missing / extra points) ----
    xs_fit_list = []
    ys_fit_list = []
    for v in data_eval_avg.values():
        y_arr = np.asarray(v, dtype=float)
        n = min(len(y_arr), len(x_base))
        if n == 0:
            continue
        xs_fit_list.append(x_base[:n])
        ys_fit_list.append(y_arr[:n])

    if not xs_fit_list:
        raise ValueError("No data points to fit.")

    xs_fit = np.concatenate(xs_fit_list)
    ys_fit = np.concatenate(ys_fit_list)

    # keep only finite pairs
    mask_finite = np.isfinite(xs_fit) & np.isfinite(ys_fit)
    xs_fit, ys_fit = xs_fit[mask_finite], ys_fit[mask_finite]

    # Fit single-rate exponential
    params = _fit_exp(xs_fit, ys_fit, x_len=len(x_base))
    residuals = ys_fit - _exp_model(xs_fit, params)
    ss_res = np.sum(residuals**2)
    rse_single = np.sqrt(ss_res / (len(ys_fit) - 3))  # residual standard error (single: 3 params)
    print(f"Single rate error: {rse_single:.5f}")
    print(f"  fit: y = {params[0]:.4f} + {params[1]:.4f} * exp(-{params[2]:.4f} * t)")

    # #######################################################################################
    # # TESTING OTHER MODEL FITS
    # #######################################################################################

    # n_obs = len(ys_fit)

    # # dual exponential
    # params_dual = _fit_dual_exp(xs_fit, ys_fit, x_len=len(x_base))
    # residuals_dual = ys_fit - _exp_model_dual(xs_fit, params_dual)
    # ss_res_dual = np.sum(residuals_dual**2)
    # rse_dual = np.sqrt(ss_res_dual / (n_obs - 5))  # residual standard error (dual: 5 params)
    # print(f"Dual rate error: {rse_dual:.5f}")
    # print(
    #     f"  fit: y = {params_dual[0]:.4f} + {params_dual[1]:.4f} * exp(-{params_dual[2]:.4f} * t)"
    #     f" + {params_dual[3]:.4f} * exp(-{params_dual[4]:.4f} * t)"
    # )

    # # see if dual exponential is significantly better than single exponential (nested F-test)
    # p_dual = 5 # assume non fixed rate
    # p_single = 3
    # df1 = p_dual - p_single
    # df2 = n_obs - p_dual
    # f_stat = ((ss_res - ss_res_dual) / df1) / (ss_res_dual / df2)
    # p_value = float(stats.f.sf(f_stat, df1, df2))
    # print(
    #     f"F({df1},{df2}) = {f_stat:.3f}, p = {p_value:.4g} -> dual exponential is "
    #     f"{'significantly' if p_value < 0.05 else 'not significantly'} better than single."
    # )

    # # constant, linear, and power fits
    # params_const = _fit_const(ys_fit)
    # ss_res_const = np.sum((ys_fit - _const_model(xs_fit, params_const)) ** 2)
    # rse_const = np.sqrt(ss_res_const / (n_obs - 1))  # residual standard error (constant: 1 param)
    # print(f"Constant rate error: {rse_const:.5f}")
    # print(f"  fit: y = {params_const[0]:.4f}")

    # params_line = _fit_line(xs_fit, ys_fit)
    # ss_res_line = np.sum((ys_fit - _line_model(xs_fit, params_line)) ** 2)
    # rse_line = np.sqrt(ss_res_line / (n_obs - 2))  # residual standard error (linear: 2 params)
    # print(f"Linear rate error: {rse_line:.5f}")
    # print(f"  fit: y = {params_line[0]:.4f} + {params_line[1]:.4f} * t")

    # params_power = _fit_power(xs_fit, ys_fit)
    # ss_res_power = np.sum((ys_fit - _power_model(xs_fit, params_power)) ** 2)
    # rse_power = np.sqrt(ss_res_power / (n_obs - 3))  # residual standard error (power: 3 params)
    # print(f"Power rate error: {rse_power:.5f}")
    # print(f"  fit: y = {params_power[0]:.4f} + {params_power[1]:.4f} * (t + {POWER_OFFSET:.1f})^(-{params_power[2]:.4f})")

    # # see if single exponential is significantly better than constant (nested F-test)
    # p_const = 1  # constant: a
    # df1 = p_single - p_const
    # df2 = n_obs - p_single
    # f_stat = ((ss_res_const - ss_res) / df1) / (ss_res / df2)
    # p_value = float(stats.f.sf(f_stat, df1, df2))
    # print(
    #     f"F({df1},{df2}) = {f_stat:.3f}, p = {p_value:.4g} -> single exponential is "
    #     f"{'significantly' if p_value < 0.05 else 'not significantly'} better than constant."
    # )
    # #######################################################################################

    if single_rate:
        A, B, k = params
    else:
        A, B, k1, C, k2 = params

    if not split:
        fig, ax = plt.subplots(dpi=150)
        ax.set_xlabel("Exposure time (minutes)")
        ax.set_ylabel("Normalized evaluation heading error")
        fig.set_figwidth(2)
        fig.set_figheight(2)
    else:
        fig, (ax, ax2) = create_split_plot(xlim=(0, 75), ax1_ylim=(0.34, 1.0), ax2_ylim=(0, 0.01), hide_yaxis=hide_yaxis)
        # ax2.set_xlabel("Exposure time (minutes)")
        fig.text(0.52, -0.015, 'Exposure time (minutes)', ha='center', va='center')
        ax2.xaxis.set_label_coords(0.53, -4)
        if not hide_yaxis:
            fig.text(-0.1, 0.5, 'Normalized evaluation heading error', ha='center', va='center', rotation='vertical')
        # ax.set_ylabel("Normalized evaluation heading error", labelpad=3)
        ax2.set_xticks([0,15,30,45,60,75])

    # Distinct color for each participant (tab10 cycles if >10)
    persons = list(data_eval_avg.keys())
    colors = cm.tab10.colors if colors is None else colors
    for idx, person in enumerate(persons):
        y_vals = np.asarray(data_eval_avg[person], dtype=float)
        n = min(len(y_vals), len(x_base))
        if n == 0:
            continue
        ax.scatter(x_base[:n], y_vals[:n], s=marker_size, alpha=1.0, color=colors[idx % 10], marker=group_marker if group_marker else "o", label=person)

    # Fitted curve
    t_fit = np.linspace(x_base.min(), x_base.max(), 200)
    ax.plot(t_fit, _exp_model(t_fit, params), linewidth=1.25, color=group_color if group_color else "black", solid_capstyle="butt")

    # bootstrap CI (clustered by participant)
    ci_lo, ci_hi, all_params = bootstrap_ci_for_group(data_eval_avg, t_fit)

    # shaded CI + curve
    ax.fill_between(t_fit, ci_lo, ci_hi, alpha=0.25, color=group_color if group_color else "black", linewidth=0)

    # Labels, grid, legend
    ax.set_title(title, fontsize=7)
    # ax.set_ylim(min(0.3,np.min(ys_fit)), max(1.1,np.max(ys_fit)))
    ax.set_yticks(np.arange(0.4, 1.001, 0.1))
    # ax.grid(visible=True, which="major", axis="y", linestyle="--", alpha=0.7)
    # ax.legend(ncol=2, fontsize=5, bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
    # Equation annotation
    if single_rate:
        eq_text = rf"$\eta = {A:.2f} + {B:.2f}e^{{-{k:.3f}t}}$"
        ax.text(0.15, 0.95, eq_text, transform=ax.transAxes, fontsize=7)
    else:
        eq_text = rf"$\eta = {A:.2f} + {B:.2f}e^{{-{k1:.3f}t}} + {C:.2f}e^{{-{k2:.3f}t}}$"
        ax.text(0.3, 0.85, eq_text, transform=ax.transAxes, fontsize=7)

    # Save if requested
    if save_path:
        fig.patch.set_visible(False)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches="tight", transparent=True)
        fig.patch.set_visible(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    if single_rate:
        return {"A": A, "B": B, "k": k, "all": all_params}
    else:
        return {"A": A, "B": B, "k1": k1, "C": C, "k2": k2, "all": all_params}

def plot_group_fits(
    data_eval_avg_dict: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    group_colors = None,
    split = False,
    hide_yaxis = False,
) -> Dict[str, float]:
    
    x_base = np.array([round(0.00 + (0.6 + 4.8 + 0.6) * i, 1) for i in range(13)], dtype=float)

    if not split:
        fig, ax = plt.subplots(dpi=150)
        ax.set_xlabel("Exposure time (minutes)")
        ax.set_ylabel("Normalized evaluation heading error")
        fig.set_figwidth(2)
        fig.set_figheight(2)
    else:
        fig, (ax, ax2) = create_split_plot(xlim=(0, 75), ax1_ylim=(0.49, 0.80), ax2_ylim=(0, 0.01), hide_yaxis=hide_yaxis, figsize=[1.9, 2.7])
        # ax2.set_xlabel("Exposure time (minutes)")
        fig.text(0.52, 0.00, 'Exposure time (minutes)', ha='center', va='center')
        ax2.xaxis.set_label_coords(0.53, -4)
        if not hide_yaxis:
            fig.text(-0.07, 0.5, 'Normalized evaluation heading error', ha='center', va='center', rotation='vertical')
        # ax.set_ylabel("Normalized evaluation heading error", labelpad=3)
        ax2.set_xticks([0,15,30,45,60,75])

    ax.set_yticks(np.arange(0.5, 0.81, 0.05))

    for group_name, data_eval_avg in data_eval_avg_dict.items():
        xs_fit_list = []
        ys_fit_list = []
        for v in data_eval_avg.values():
            y_arr = np.asarray(v, dtype=float)
            n = min(len(y_arr), len(x_base))
            if n == 0:
                continue
            xs_fit_list.append(x_base[:n])
            ys_fit_list.append(y_arr[:n])

        xs_fit = np.concatenate(xs_fit_list)
        ys_fit = np.concatenate(ys_fit_list)

        # keep only finite pairs
        mask_finite = np.isfinite(xs_fit) & np.isfinite(ys_fit)
        xs_fit, ys_fit = xs_fit[mask_finite], ys_fit[mask_finite]

        # Fit parameters
        params = _fit_exp(xs_fit, ys_fit, x_len=len(x_base))

        # Fitted curve
        t_fit = np.linspace(x_base.min(), x_base.max(), 200)
        group_color = group_colors[group_name] if group_colors else "black"
        ax.plot(t_fit, _exp_model(t_fit, params), linewidth=1.25, color=group_color, label=group_name, solid_capstyle="butt")

        # bootstrap CI (clustered by participant)
        ci_lo, ci_hi, all_params = bootstrap_ci_for_group(data_eval_avg, t_fit)

        ax.fill_between(t_fit, ci_lo, ci_hi, alpha=0.125, color=group_color, linewidth=0)

    # ax.legend(ncol=1, frameon=False, loc='upper right')

    if save_path:
        fig.patch.set_visible(False)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches="tight", transparent=True)
        fig.patch.set_visible(True)

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
def create_split_plot(xlim, ax1_ylim, ax2_ylim=(0, 0.0001), hide_yaxis=False, figsize=[1.37,2.2]):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [30, 1]}, figsize=figsize, dpi=150)
    fig.subplots_adjust(hspace=0.05) # Space between axes

    # Hide spines/ticks for ax1
    ax1.spines.bottom.set_visible(False)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    if hide_yaxis:
        ax1.spines['left'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    else:
        # Move the y-axis left (separating it from the x-axis)
        y_spine_pos = -0.03
        ax1.spines['left'].set_position(('axes', y_spine_pos))
        ax2.spines['left'].set_position(('axes', y_spine_pos))
        
    # Move the x-axis down
    x_spine_pos = -0.8
    ax2.spines['bottom'].set_position(('axes', x_spine_pos))

    # Apply a slight padding to the left limit so markers at x=0 aren't cut by the axes bounding box
    x_pad = (xlim[1] - xlim[0]) * 0.02
    ax1.set_xlim(xlim[0] - x_pad, xlim[1])
    ax1.set_ylim(ax1_ylim)
    ax2.set_ylim(ax2_ylim)
    
    if hide_yaxis:
        ax2.set_yticks([])
    else:
        ax2.set_yticks([0])
    
    # Restrict the actual drawn spine line to visually start at `0`
    ax2.spines['bottom'].set_bounds(xlim[0], xlim[1])

    if not hide_yaxis:
        # Slanted lines
        d = 0.5  # slope of slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
                    linestyle="none", color='k', mec='k', mew=0.5, clip_on=False)
        # Move the slanted lines with the y-axis
        ax1.plot([y_spine_pos], [0], transform=ax1.transAxes, **kwargs)
        ax2.plot([y_spine_pos], [1], transform=ax2.transAxes, **kwargs)

    # Move axis labels closer to the axis
    ax1.tick_params(axis='x', pad=2)
    ax2.tick_params(axis='x', pad=2)
    ax1.tick_params(axis='y', pad=1.5)
    ax2.tick_params(axis='y', pad=1.5)

    return fig, (ax1, ax2)

def create_unsplit_plot(xlim, ax1_ylim, figsize=[1.4,2.2]):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Move the y-axis left (separating it from the x-axis)
    y_spine_pos = -0.04
    ax.spines['left'].set_position(('axes', y_spine_pos))
        
    # Move the x-axis down
    x_spine_pos = -0.03
    ax.spines['bottom'].set_position(('axes', x_spine_pos))

    # Apply a slight padding to the left limit so markers at x=0 aren't cut by the axes bounding box
    x_pad = (xlim[1] - xlim[0]) * 0.02
    ax.set_xlim(xlim[0] - x_pad, xlim[1])
    ax.set_ylim(ax1_ylim)

    # Move axis labels closer to the axis
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=1.5)

    return fig, ax

def draw_bracket(x1, x2, y, pval, ax=None):
    if ax is not None:
        ax1 = ax
    else:
        ax1 = plt.gca()

    ymin, ymax = ax1.get_ylim()

    # Bracket
    h = 0.01 * (ymax - ymin)
    pvaltext = ''
    if pval < 1e-3:
        pvaltext = '***'
    elif pval < 1e-2:
        pvaltext = '**'
    elif pval < 5e-2:
        pvaltext = '*'
    x1 = x1 + 1
    x2 = x2 + 1
    ax1.plot([x1, x1, x2, x2], [y-h, y, y, y-h], color='black', linewidth=0.5)
    ax1.text((x1+x2)/2, y - 0.01*(ymax - ymin), pvaltext, ha='center', va='baseline', color='black', fontsize=7)
    if pvaltext == '':
        return False
    return True

def create_split_plot_xy():
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row',
                                gridspec_kw={'height_ratios': [30, 1], 'width_ratios': [1, 30]}, 
                                figsize=[2.7,2.7], dpi=120)
    # ax1 is top-left, ax2 is top-right, ax3 is bottom-left, ax4 is bottom-right
    (ax1, ax2), (ax3, ax4) = axes
    fig.subplots_adjust(hspace=0.05, wspace=0.05) # Space between axes

    # y axis
    # ax1.set_ylim(1.5,7)
    ax3.set_ylim(0,0.001)
    # x axis
    ax3.set_xlim(0,0.001)
    # ax4.set_xlim(1.5,8)

    # ax3 ticks
    ax3.set_xticks([0])
    ax3.set_yticks([0])

    # Hide spines/ticks
    ax1.spines.bottom.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.bottom.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax3.spines.right.set_visible(False)
    ax4.spines.top.set_visible(False)
    ax4.spines.left.set_visible(False)

    ax1.tick_params(bottom=False, labelbottom=False, right=False)
    ax2.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False, right=False)
    ax3.tick_params(top=False, right=False)
    ax4.tick_params(top=False, left=False, labelleft=False, right=False)

    d = 1  # slope of slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
                linestyle="none", color='k', mec='k', mew=0.5, clip_on=False)

    # y-axis break (left spine)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax3.plot([0], [1], transform=ax3.transAxes, **kwargs)

    # x-axis break (bottom spine)
    ax3.plot([1], [0], transform=ax3.transAxes, **kwargs)
    ax4.plot([0], [0], transform=ax4.transAxes, **kwargs)

    return fig, (ax1, ax2, ax3, ax4)