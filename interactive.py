#!/usr/bin/env python3
"""
Interactive Phased Array Beam Simulation Tool
=============================================
Gradio-based GUI for exploring ultrasonic phased array beamsteering.
Adjust physical parameters, run k-Wave simulations, and visualise results.

Usage:
    python interactive.py              # launches in browser
    python interactive.py --share      # creates a public URL
"""

import argparse
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gradio as gr

from simulation import (
    run_simulation, focusing_phases,
    F0, C0, K0, LAM, P_REF,
)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers (adapted from visualizer.py)
# ══════════════════════════════════════════════════════════════════════════════

CMAP_SPL   = 'inferno'
FOCAL_CLR  = '#00FFAA'
ELEM_CLR   = '#44AAFF'
ARRAY1_CLR = '#FF6B35'
ARRAY2_CLR = '#4FC3F7'
COMB_CLR   = '#FFFFFF'


def cm(arr):
    return np.asarray(arr) * 100.0


def _save_fig(fig):
    """Save figure to a temporary PNG and return the path."""
    f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(f.name, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return f.name


def _add_colourbar(fig, ax, im, label, pad=0.05):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=pad)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, labelpad=4)
    cb.ax.tick_params(labelsize=8)
    return cb


def _mark_focal(ax, xf_cm, yf_cm, label=True):
    kw = dict(color=FOCAL_CLR, lw=0.8, ls='--', alpha=0.9)
    ax.axvline(xf_cm, **kw)
    ax.axhline(yf_cm, **kw)
    ax.scatter(xf_cm, yf_cm, s=60, c=FOCAL_CLR,
               marker='*', zorder=5, edgecolors='k', linewidths=0.4)
    if label:
        ax.text(xf_cm + 0.4, yf_cm + 0.4, "focal\npoint",
                color=FOCAL_CLR, fontsize=7, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))


def _mark_elements(ax, xe_cm, ye_cm, color=ELEM_CLR):
    ax.scatter(xe_cm, ye_cm, s=18, c=color, marker='s',
               zorder=5, edgecolors='k', linewidths=0.3)


def _spl_imshow(ax, x_cm, y_cm, spl_field, vmin, vmax):
    im = ax.imshow(spl_field.T, origin='lower',
                   extent=[x_cm[0], x_cm[-1], y_cm[0], y_cm[-1]],
                   aspect='equal', cmap=CMAP_SPL, vmin=vmin, vmax=vmax,
                   interpolation='bilinear')
    ax.set_xlabel("x  [cm]")
    ax.set_ylabel("y  [cm]")
    return im


# ══════════════════════════════════════════════════════════════════════════════
# Figure generators
# ══════════════════════════════════════════════════════════════════════════════

def make_field_comparison(data):
    """Unfocused vs focused combined field — side by side."""
    plt.style.use('dark_background')

    x_cm = cm(data['x_vec'])
    y_cm = cm(data['y_vec'])
    xf_cm = cm(float(data['x_focal']))
    yf_cm = cm(float(data['y_focal']))
    xe1_cm, ye1_cm = cm(data['x1']), cm(data['y1'])
    xe2_cm, ye2_cm = cm(data['x2']), cm(data['y2'])

    spl_unfoc = data['spl_comb_unfoc']
    spl_foc = data['spl_comb_foc']

    vmax = float(np.nanmax(spl_foc))
    vmin = vmax - 50.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.patch.set_facecolor('#1A1A1A')
    fig.suptitle(
        "Effect of Beamsteering — Combined Field\n"
        f"Focal point at ({xf_cm:.0f}, {yf_cm:.0f}) cm  |  "
        "k-Wave FDTD  |  40 kHz CW",
        fontsize=11, color='#EEEEEE')

    for ax, field, title in [
        (axes[0], spl_unfoc, "Unfocused (all phases = 0)"),
        (axes[1], spl_foc,   "Focused (delay-and-sum beamsteering)"),
    ]:
        im = _spl_imshow(ax, x_cm, y_cm, field, vmin, vmax)
        ax.set_title(title, color='#EEEEEE')
        _mark_focal(ax, xf_cm, yf_cm)
        _mark_elements(ax, xe1_cm, ye1_cm, ARRAY1_CLR)
        _mark_elements(ax, xe2_cm, ye2_cm, ARRAY2_CLR)
        _add_colourbar(fig, ax, im, "SPL [dB]")
        peak = np.nanmax(field)
        ax.text(0.02, 0.98, f"peak {peak:.1f} dB", transform=ax.transAxes,
                fontsize=8, va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc='#00000077', ec='none'))

    # Annotate gain
    ix_f, iy_f = int(data['ix_f']), int(data['iy_f'])
    gain = spl_foc[ix_f, iy_f] - spl_unfoc[ix_f, iy_f]
    axes[1].text(0.02, 0.88,
                 f"Focusing gain: {gain:+.1f} dB",
                 transform=axes[1].transAxes, fontsize=9, va='top', ha='left',
                 color=FOCAL_CLR, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#00000088', ec='none'))

    return _save_fig(fig)


def make_phase_map(data):
    """NxN phase heatmap for each array."""
    plt.style.use('dark_background')

    N = int(data['N'])
    xf = float(data['x_focal'])
    yf = float(data['y_focal'])
    k = K0
    d_m = float(data['d'])
    z_offsets = (np.arange(N) - (N - 1) / 2) * d_m

    def compute_NxN_phases(xe_arr, ye_arr):
        phi_grid = np.zeros((N, N))
        r_all = []
        for z_j in z_offsets:
            for i in range(N):
                r = np.sqrt((xf - xe_arr[i])**2 + (yf - ye_arr[i])**2 + z_j**2)
                r_all.append(r)
        r_min = min(r_all)
        for j, z_j in enumerate(z_offsets):
            for i in range(N):
                r = np.sqrt((xf - xe_arr[i])**2 + (yf - ye_arr[i])**2 + z_j**2)
                phi_grid[i, j] = k * (r - r_min)
        return phi_grid

    phi1_grid = compute_NxN_phases(data['x1'], data['y1'])
    phi2_grid = compute_NxN_phases(data['x2'], data['y2'])
    phi_max = max(phi1_grid.max(), phi2_grid.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    fig.patch.set_facecolor('#1A1A1A')
    fig.suptitle(
        r"Phase Law:  $\varphi_{ij} = k(r_{ij} - r_{\min})$"
        f"\n{N}x{N} element arrays  |  "
        f"pitch = {d_m*1e3:.2f} mm  |  focal at "
        f"({xf*100:.0f}, {yf*100:.0f}) cm",
        fontsize=11, color='#EEEEEE')

    y_labels = [f"{yi:.1f}" for yi in cm(data['y1'])]
    z_labels = [f"{zi:.1f}" for zi in cm(z_offsets)]

    for ax, phi_grid, label in [
        (axes[0], phi1_grid, "Array 1 (upper)"),
        (axes[1], phi2_grid, "Array 2 (lower)"),
    ]:
        im = ax.imshow(phi_grid, cmap='viridis', vmin=0, vmax=phi_max,
                        origin='lower', aspect='equal')
        for i in range(N):
            for j in range(N):
                val = phi_grid[i, j]
                text_color = 'white' if val < phi_max * 0.6 else 'black'
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                        fontsize=max(5, 8 - N // 4), color=text_color,
                        fontweight='bold')

        ax.set_xticks(range(N))
        ax.set_xticklabels(z_labels, fontsize=max(5, 8 - N // 4))
        ax.set_yticks(range(N))
        ax.set_yticklabels(y_labels, fontsize=max(5, 8 - N // 4))
        ax.set_xlabel("z (out-of-plane)  [cm]")
        ax.set_ylabel("y (in-plane)  [cm]")
        ax.set_title(label, color='#EEEEEE')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"Phase $\varphi_{ij}$  [rad]", fontsize=9)
        cb.ax.tick_params(labelsize=7)

    return _save_fig(fig)


def make_beam_profiles(data):
    """Axial and lateral SPL profiles through the focal point."""
    plt.style.use('dark_background')

    x_cm = cm(data['x_vec'])
    y_cm = cm(data['y_vec'])
    xf_cm = cm(float(data['x_focal']))
    yf_cm = cm(float(data['y_focal']))
    ix_f = int(data['ix_f'])
    iy_f = int(data['iy_f'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    fig.patch.set_facecolor('#1A1A1A')
    fig.suptitle("Beam Profiles through the Focal Point",
                 fontsize=12, fontweight='bold', color='#EEEEEE')

    # Axial profile (y = y_focal)
    ax = axes[0]
    ax.plot(x_cm, data['spl_comb_foc'][:, iy_f], color=COMB_CLR, lw=1.8,
            label="Focused combined")
    ax.plot(x_cm, data['spl_comb_unfoc'][:, iy_f], color='#888888', lw=1.0,
            ls='-.', alpha=0.7, label="Unfocused combined")
    peak_c = float(data['spl_comb_foc'][ix_f, iy_f])
    ax.axhline(peak_c - 3, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(peak_c - 6, color='grey', lw=0.6, ls=':', alpha=0.6)
    ax.text(x_cm[0] + 0.3, peak_c - 3 + 0.3, "-3 dB", fontsize=7, color='grey')
    ax.text(x_cm[0] + 0.3, peak_c - 6 + 0.3, "-6 dB", fontsize=7, color='grey')
    ax.axvline(xf_cm, color=FOCAL_CLR, lw=0.8, ls='--', alpha=0.7, label="focal x")
    ax.set_xlabel("x  [cm]")
    ax.set_ylabel("SPL  [dB re 20 \u00b5Pa]")
    ax.set_title(f"Axial profile  (y = {yf_cm:.0f} cm)", color='#EEEEEE')
    ax.legend(fontsize=8, loc='lower right')

    # Lateral profile (x = x_focal)
    ax = axes[1]
    ax.plot(data['spl_comb_foc'][ix_f, :], y_cm, color=COMB_CLR, lw=1.8,
            label="Focused combined")
    ax.plot(data['spl_comb_unfoc'][ix_f, :], y_cm, color='#888888', lw=1.0,
            ls='-.', alpha=0.7, label="Unfocused combined")
    ax.axvline(peak_c - 3, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.axvline(peak_c - 6, color='grey', lw=0.6, ls=':', alpha=0.6)
    ax.axhline(yf_cm, color=FOCAL_CLR, lw=0.8, ls='--', alpha=0.7, label="focal y")
    ax.set_xlabel("SPL  [dB re 20 \u00b5Pa]")
    ax.set_ylabel("y  [cm]")
    ax.set_title(f"Lateral profile  (x = {xf_cm:.0f} cm)", color='#EEEEEE')
    ax.legend(fontsize=8, loc='lower right')

    for a in axes:
        a.set_facecolor('#111111')
        a.grid(True, lw=0.3, alpha=0.4)
        a.spines[:].set_color('#444444')
        a.tick_params(colors='#CCCCCC')
        a.xaxis.label.set_color('#CCCCCC')
        a.yaxis.label.set_color('#CCCCCC')

    return _save_fig(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics and analysis
# ══════════════════════════════════════════════════════════════════════════════

def _beam_width_3dB(profile, coord):
    """Estimate -3 dB width from a 1D SPL profile."""
    peak = np.max(profile)
    threshold = peak - 3.0
    above = profile >= threshold
    if not np.any(above):
        return None
    indices = np.where(above)[0]
    return abs(coord[indices[-1]] - coord[indices[0]])


def _grating_lobe_check(d_factor, theta_steer_deg):
    """Check if grating lobes exist in visible space."""
    d_over_lam = d_factor
    sin_steer = np.sin(np.radians(theta_steer_deg))
    # First grating lobe: sin(theta_g) = sin(theta_s) + lambda/d = sin(theta_s) + 1/d_factor
    sin_grating = sin_steer + 1.0 / d_over_lam
    if abs(sin_grating) <= 1.0:
        return True, np.degrees(np.arcsin(min(1.0, abs(sin_grating))))
    sin_grating_neg = sin_steer - 1.0 / d_over_lam
    if abs(sin_grating_neg) <= 1.0:
        return True, np.degrees(np.arcsin(min(1.0, abs(sin_grating_neg))))
    return False, None


def make_metrics(data):
    """Build a markdown summary of key results."""
    ix_f = int(data['ix_f'])
    iy_f = int(data['iy_f'])
    N = int(data['N'])
    d = float(data['d'])
    d_factor = float(data['d_factor'])
    A_elem = float(data['A_elem'])
    x_focal = float(data['x_focal'])
    y_focal = float(data['y_focal'])
    array_sep = float(data['array_sep'])

    spl_unfoc = float(data['spl_comb_unfoc'][ix_f, iy_f])
    spl_foc = float(data['spl_comb_foc'][ix_f, iy_f])
    gain = spl_foc - spl_unfoc
    p_peak = float(data['p_max_comb_foc'][ix_f, iy_f])

    x_cm = cm(data['x_vec'])
    y_cm = cm(data['y_vec'])

    # Beam widths
    axial_profile = data['spl_comb_foc'][:, iy_f]
    lateral_profile = data['spl_comb_foc'][ix_f, :]
    bw_axial = _beam_width_3dB(axial_profile, x_cm)
    bw_lateral = _beam_width_3dB(lateral_profile, y_cm)

    # Steering angle (from Array 1 centre to focal point)
    y_c1 = float(data['y_c1'])
    theta_steer = np.degrees(np.arctan2(abs(y_c1 - y_focal), x_focal))

    # Grating lobe check
    has_grating, gl_angle = _grating_lobe_check(d_factor, theta_steer)

    # Phase delays summary
    phi1 = data['phi1_focus']
    phi2 = data['phi2_focus']

    md = "## Simulation Results\n\n"

    # Main metrics table
    md += "| Metric | Value |\n|---|---|\n"
    md += f"| Focused SPL (combined) | **{spl_foc:.1f} dB** |\n"
    md += f"| Unfocused SPL (combined) | {spl_unfoc:.1f} dB |\n"
    md += f"| **Focusing gain** | **{gain:+.1f} dB** |\n"
    md += f"| Peak pressure at focus | {p_peak:.3f} Pa |\n"
    if bw_axial is not None:
        md += f"| -3 dB beam width (axial) | {bw_axial:.1f} cm |\n"
    if bw_lateral is not None:
        md += f"| -3 dB beam width (lateral) | {bw_lateral:.1f} cm |\n"
    md += f"| Steering angle | {theta_steer:.1f} deg |\n"
    md += "\n"

    # Setup summary
    md += "### Configuration\n\n"
    md += "| Parameter | Value |\n|---|---|\n"
    md += f"| Frequency | {F0/1e3:.0f} kHz |\n"
    md += f"| Wavelength | {LAM*1e3:.2f} mm |\n"
    md += f"| Elements per array | {N} ({N}x{N} physical) |\n"
    md += f"| Element spacing | {d*1e3:.2f} mm ({d_factor:.2f} x lambda) |\n"
    md += f"| Array aperture | {(N-1)*d*1e3:.1f} mm |\n"
    md += f"| Array separation | {array_sep*100:.0f} cm |\n"
    md += f"| Source amplitude | {A_elem:.0f} Pa/element |\n"
    md += f"| Focal point | ({x_focal*100:.0f}, {y_focal*100:.0f}) cm |\n"
    md += "\n"

    # Grating lobe warning
    if has_grating:
        md += (f"> **Warning: Grating lobes detected.** Element spacing "
               f"({d_factor:.2f} x lambda) exceeds lambda/2 at this steering "
               f"angle ({theta_steer:.0f} deg). Parasitic beam near "
               f"{gl_angle:.0f} deg. Consider reducing element spacing.\n\n")

    # Optimal phase delays
    md += "### Optimal Phase Delays (delay-and-sum)\n\n"
    md += "**Array 1 (upper):**\n```\n"
    for i, (yi, phi_i, ri) in enumerate(zip(data['y1'], phi1, data['r1'])):
        md += f"  elem {i+1}:  y = {yi*100:+6.2f} cm   r = {ri*100:.2f} cm   phi = {phi_i:.3f} rad\n"
    md += "```\n\n"
    md += "**Array 2 (lower):**\n```\n"
    for i, (yi, phi_i, ri) in enumerate(zip(data['y2'], phi2, data['r2'])):
        md += f"  elem {i+1}:  y = {yi*100:+6.2f} cm   r = {ri*100:.2f} cm   phi = {phi_i:.3f} rad\n"
    md += "```\n"

    return md


# ══════════════════════════════════════════════════════════════════════════════
# Main simulation callback
# ══════════════════════════════════════════════════════════════════════════════

def run_interactive(N, sep_cm, amplitude, xf_cm, yf_cm, spacing_factor,
                    progress=gr.Progress(track_tqdm=False)):
    """Run the simulation with user-specified parameters and return results."""
    sep = sep_cm / 100.0
    xf = xf_cm / 100.0
    yf = yf_cm / 100.0

    progress(0.0, desc="Setting up simulation grid...")

    def progress_cb(step, total, desc):
        progress(step / total * 0.85, desc=f"k-Wave solve {step}/{total}: {desc}")

    results = run_simulation(
        N=int(N),
        array_sep=sep,
        x_focal=xf,
        y_focal=yf,
        A_elem=amplitude,
        d_factor=spacing_factor,
        verbose=False,
        progress_cb=progress_cb,
        run_individual=False,
    )

    progress(0.85, desc="Generating figures...")

    metrics = make_metrics(results)
    field_fig = make_field_comparison(results)

    progress(0.90, desc="Generating phase map...")
    phase_fig = make_phase_map(results)

    progress(0.95, desc="Generating beam profiles...")
    profile_fig = make_beam_profiles(results)

    progress(1.0, desc="Done")
    return metrics, field_fig, phase_fig, profile_fig


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(
        title="Phased Array Beam Simulation",
    ) as app:

        gr.Markdown(
            "# Ultrasonic Phased Array Beam Simulation\n"
            "Adjust the physical parameters below, then click **Run Simulation** "
            "to compute the beamsteering solution and visualise the acoustic field. "
            "The simulation solves the full wave equation using the k-Wave "
            "pseudospectral time-domain method.",
        )

        with gr.Row():
            # ── Left column: controls ──
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Array Parameters")
                n_slider = gr.Slider(
                    minimum=4, maximum=20, value=8, step=2,
                    label="Elements per array (N)",
                    info="Each array is NxN; 2D simulation uses N-element cross-section",
                )
                sep_slider = gr.Slider(
                    minimum=10, maximum=60, value=30, step=2,
                    label="Array separation [cm]",
                    info="Vertical distance between the two array centres",
                )
                spacing_slider = gr.Slider(
                    minimum=0.25, maximum=1.0, value=0.5, step=0.05,
                    label="Element spacing [x lambda]",
                    info="0.50 = lambda/2 (Nyquist). Above 0.5 may produce grating lobes",
                )

                gr.Markdown("### Source & Focus")
                amp_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Source amplitude [Pa per element]",
                )
                xf_slider = gr.Slider(
                    minimum=5, maximum=45, value=25, step=1,
                    label="Focal point X [cm]",
                    info="Distance downrange from the arrays",
                )
                yf_slider = gr.Slider(
                    minimum=-20, maximum=20, value=0, step=1,
                    label="Focal point Y [cm]",
                    info="0 = midline between arrays",
                )

                run_btn = gr.Button(
                    "Run Simulation",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown(
                    "*Each run solves 2 k-Wave simulations (unfocused + focused). "
                    "Expect ~30-90 seconds depending on grid size.*"
                )

            # ── Right column: results ──
            with gr.Column(scale=3):
                metrics_output = gr.Markdown(
                    value="### Results will appear here after running the simulation.",
                )
                with gr.Tabs():
                    with gr.Tab("Field Comparison"):
                        field_output = gr.Image(
                            label="Unfocused vs Focused combined field",
                        )
                    with gr.Tab("Phase Map"):
                        phase_output = gr.Image(
                            label="Per-element phase delays (NxN array)",
                        )
                    with gr.Tab("Beam Profiles"):
                        profile_output = gr.Image(
                            label="Axial and lateral SPL cuts through focal point",
                        )

        run_btn.click(
            fn=run_interactive,
            inputs=[n_slider, sep_slider, amp_slider,
                    xf_slider, yf_slider, spacing_slider],
            outputs=[metrics_output, field_output, phase_output, profile_output],
        )

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive phased array simulation")
    parser.add_argument("--share", action="store_true",
                        help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                        help="local port (default 7860)")
    args = parser.parse_args()

    app = build_app()
    app.launch(share=args.share, server_port=args.port)
