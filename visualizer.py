#!/usr/bin/env python3
"""
Visualizer for the k-Wave phased array simulation.

Produces:
  1. Unfocused field comparison: Array 1 | Array 2 | Combined  (no beamsteering)
  2. Focused field comparison:   Array 1 | Array 2 | Combined  (beamsteered)
  3. Before/after comparison:    Unfocused combined | Focused combined
  4. Phase map: per-element delay-and-sum phases
  5. Beam profiles: axial and lateral cuts through the focal point

Usage
-----
    python simulation.py       # produces results.npz
    python visualizer.py       # generates all figures
"""

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Aesthetic settings ────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.linewidth':    0.8,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'figure.dpi':        150,
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
    'image.interpolation': 'bilinear',
})

# ── Colour choices ────────────────────────────────────────────────────────────
CMAP_SPL   = 'inferno'
FOCAL_CLR  = '#00FFAA'
ELEM_CLR   = '#44AAFF'
ARRAY1_CLR = '#FF6B35'
ARRAY2_CLR = '#4FC3F7'
COMB_CLR   = '#FFFFFF'


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def cm(arr):
    return np.asarray(arr) * 100.0

def add_colourbar(fig, ax, im, label, pad=0.05):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=pad)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, labelpad=4)
    cb.ax.tick_params(labelsize=8)
    return cb

def mark_focal(ax, xf_cm, yf_cm, label=True):
    kw = dict(color=FOCAL_CLR, lw=0.8, ls='--', alpha=0.9)
    ax.axvline(xf_cm, **kw)
    ax.axhline(yf_cm, **kw)
    ax.scatter(xf_cm, yf_cm, s=60, c=FOCAL_CLR,
               marker='*', zorder=5, edgecolors='k', linewidths=0.4)
    if label:
        ax.text(xf_cm + 0.4, yf_cm + 0.4, "focal\npoint",
                color=FOCAL_CLR, fontsize=7, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))

def mark_elements(ax, xe_cm, ye_cm, color=ELEM_CLR, label=None):
    ax.scatter(xe_cm, ye_cm, s=18, c=color, marker='s',
               zorder=5, edgecolors='k', linewidths=0.3, label=label)

def spl_imshow(ax, x_cm, y_cm, spl_field, vmin=60.0, vmax=None):
    """Plot SPL field. Note: k-Wave returns [Nx, Ny] → transpose for imshow."""
    if vmax is None:
        vmax = float(np.nanmax(spl_field))
    # spl_field has shape [Nx, Ny] where x is first axis
    # For plotting: x → horizontal, y → vertical → transpose
    im = ax.imshow(spl_field.T, origin='lower',
                   extent=[x_cm[0], x_cm[-1], y_cm[0], y_cm[-1]],
                   aspect='equal', cmap=CMAP_SPL, vmin=vmin, vmax=vmax,
                   interpolation='bilinear')
    ax.set_xlabel("x  [cm]")
    ax.set_ylabel("y  [cm]")
    return im, vmax


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Unfocused field comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_unfocused_fields(data, show=False):
    x_cm = cm(data['x_vec'])
    y_cm = cm(data['y_vec'])
    xf_cm = cm(float(data['x_focal']))
    yf_cm = cm(float(data['y_focal']))

    spl1 = data['spl_arr1_unfoc']
    spl2 = data['spl_arr2_unfoc']
    splc = data['spl_comb_unfoc']

    vmax = float(np.nanmax(splc))
    vmin = vmax - 50.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), constrained_layout=True)
    fig.suptitle(
        "UNFOCUSED — No Beamsteering (all phases = 0)\n"
        "k-Wave time-domain FDTD  |  40 kHz CW  |  SPL [dB re 20 $\\mu$Pa]",
        fontsize=11)

    N = int(data['N'])
    titles = [
        f"Array 1  ({N} elements, y > 0)",
        f"Array 2  ({N} elements, y < 0)",
        "Combined  (both arrays)",
    ]
    fields = [spl1, spl2, splc]

    xe1_cm, ye1_cm = cm(data['x1']), cm(data['y1'])
    xe2_cm, ye2_cm = cm(data['x2']), cm(data['y2'])

    for i, (ax, title, field) in enumerate(zip(axes, titles, fields)):
        im, _ = spl_imshow(ax, x_cm, y_cm, field, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        mark_focal(ax, xf_cm, yf_cm, label=(i == 2))

        if i == 0:
            mark_elements(ax, xe1_cm, ye1_cm, ARRAY1_CLR)
        elif i == 1:
            mark_elements(ax, xe2_cm, ye2_cm, ARRAY2_CLR)
        else:
            mark_elements(ax, xe1_cm, ye1_cm, ARRAY1_CLR)
            mark_elements(ax, xe2_cm, ye2_cm, ARRAY2_CLR)

        add_colourbar(fig, ax, im, "SPL [dB]")
        peak = np.nanmax(field)
        ax.text(0.02, 0.98, f"peak {peak:.1f} dB", transform=ax.transAxes,
                fontsize=8, va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc='#00000077', ec='none'))

    out = "unfocused_fields.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved -> {out}")
    if show: plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Focused field comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_focused_fields(data, show=False):
    x_cm = cm(data['x_vec'])
    y_cm = cm(data['y_vec'])
    xf_cm = cm(float(data['x_focal']))
    yf_cm = cm(float(data['y_focal']))

    spl1 = data['spl_arr1_foc']
    spl2 = data['spl_arr2_foc']
    splc = data['spl_comb_foc']

    vmax = float(np.nanmax(splc))
    vmin = vmax - 50.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), constrained_layout=True)
    fig.suptitle(
        "FOCUSED — Delay-and-Sum Beamsteering to Common Focal Point\n"
        "k-Wave time-domain FDTD  |  40 kHz CW  |  SPL [dB re 20 $\\mu$Pa]",
        fontsize=11)

    N = int(data['N'])
    titles = [
        f"Array 1  (focused, {N} elements)",
        f"Array 2  (focused, {N} elements)",
        "Combined  (coherent superposition)",
    ]
    fields = [spl1, spl2, splc]

    xe1_cm, ye1_cm = cm(data['x1']), cm(data['y1'])
    xe2_cm, ye2_cm = cm(data['x2']), cm(data['y2'])

    for i, (ax, title, field) in enumerate(zip(axes, titles, fields)):
        im, _ = spl_imshow(ax, x_cm, y_cm, field, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        mark_focal(ax, xf_cm, yf_cm, label=(i == 2))

        if i == 0:
            mark_elements(ax, xe1_cm, ye1_cm, ARRAY1_CLR)
        elif i == 1:
            mark_elements(ax, xe2_cm, ye2_cm, ARRAY2_CLR)
        else:
            mark_elements(ax, xe1_cm, ye1_cm, ARRAY1_CLR)
            mark_elements(ax, xe2_cm, ye2_cm, ARRAY2_CLR)

        add_colourbar(fig, ax, im, "SPL [dB]")
        peak = np.nanmax(field)
        ax.text(0.02, 0.98, f"peak {peak:.1f} dB", transform=ax.transAxes,
                fontsize=8, va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc='#00000077', ec='none'))

    out = "focused_fields.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved -> {out}")
    if show: plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Before/After focusing comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_focusing_comparison(data, show=False):
    x_cm = cm(data['x_vec'])
    y_cm = cm(data['y_vec'])
    xf_cm = cm(float(data['x_focal']))
    yf_cm = cm(float(data['y_focal']))

    spl_unfoc = data['spl_comb_unfoc']
    spl_foc = data['spl_comb_foc']

    # Use the focused field's peak for colour scale
    vmax = float(np.nanmax(spl_foc))
    vmin = vmax - 50.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    fig.suptitle(
        "Effect of Beamsteering — Combined Field\n"
        f"Focal point at ({xf_cm:.0f}, {yf_cm:.0f}) cm  |  "
        "k-Wave FDTD  |  40 kHz CW",
        fontsize=11)

    xe1_cm, ye1_cm = cm(data['x1']), cm(data['y1'])
    xe2_cm, ye2_cm = cm(data['x2']), cm(data['y2'])

    for ax, field, title in [
        (axes[0], spl_unfoc, "BEFORE — Unfocused (all phases = 0)"),
        (axes[1], spl_foc,   "AFTER — Delay-and-sum beamsteering"),
    ]:
        im, _ = spl_imshow(ax, x_cm, y_cm, field, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        mark_focal(ax, xf_cm, yf_cm)
        mark_elements(ax, xe1_cm, ye1_cm, ARRAY1_CLR)
        mark_elements(ax, xe2_cm, ye2_cm, ARRAY2_CLR)
        add_colourbar(fig, ax, im, "SPL [dB]")

        peak = np.nanmax(field)
        ax.text(0.02, 0.98, f"peak {peak:.1f} dB", transform=ax.transAxes,
                fontsize=8, va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc='#00000077', ec='none'))

    # Annotate gain
    ix_f = int(data['ix_f'])
    iy_f = int(data['iy_f'])
    gain = spl_foc[ix_f, iy_f] - spl_unfoc[ix_f, iy_f]
    axes[1].text(0.02, 0.88,
                 f"Focusing gain at target: {gain:+.1f} dB",
                 transform=axes[1].transAxes, fontsize=9, va='top', ha='left',
                 color=FOCAL_CLR, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#00000088', ec='none'))

    out = "focusing_comparison.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved -> {out}")
    if show: plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Phase map
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_map(data, show=False):
    phi1 = data['phi1_focus']
    phi2 = data['phi2_focus']
    N = int(data['N'])
    d_cm = cm(float(data['d']))
    xf = float(data['x_focal'])
    yf = float(data['y_focal'])
    y_c1 = float(data['y_c1'])
    y_c2 = float(data['y_c2'])
    k = 2.0 * np.pi * 40_000.0 / 343.0

    # Build full NxN phase grids for each array.
    # In 2D we model one row (along y), but the physical array is NxN.
    # The z-axis (out-of-plane) elements share the same y-offset as their row
    # but have an additional z-offset that increases distance to the focal point.
    d_m = float(data['d'])
    z_offsets = (np.arange(N) - (N - 1) / 2) * d_m  # out-of-plane positions

    def compute_NxN_phases(xe_arr, ye_arr, xf, yf):
        """Compute NxN phase grid: rows = y-elements (in-plane), cols = z-elements."""
        phi_grid = np.zeros((N, N))
        r_all = []
        for j, z_j in enumerate(z_offsets):
            for i in range(N):
                r = np.sqrt((xf - xe_arr[i])**2 + (yf - ye_arr[i])**2 + z_j**2)
                r_all.append(r)
        r_min = min(r_all)
        for j, z_j in enumerate(z_offsets):
            for i in range(N):
                r = np.sqrt((xf - xe_arr[i])**2 + (yf - ye_arr[i])**2 + z_j**2)
                phi_grid[i, j] = k * (r - r_min)
        return phi_grid

    phi1_grid = compute_NxN_phases(data['x1'], data['y1'], xf, yf)
    phi2_grid = compute_NxN_phases(data['x2'], data['y2'], xf, yf)

    phi_max = max(phi1_grid.max(), phi2_grid.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    fig.suptitle(
        r"Delay-and-Sum Phase Law:  $\varphi_{ij} = k(r_{ij} - r_{\min})$"
        f"\n{N}$\\times${N} element arrays  |  "
        r"$\lambda/2$ pitch  |  focal point at "
        f"({xf*100:.0f}, {yf*100:.0f}) cm",
        fontsize=11, fontweight='bold')

    y_labels = [f"{yi:.1f}" for yi in cm(data['y1'])]
    z_labels = [f"{zi:.1f}" for zi in cm(z_offsets)]

    for ax, phi_grid, label in [
        (axes[0], phi1_grid, "Array 1 (upper)"),
        (axes[1], phi2_grid, "Array 2 (lower)"),
    ]:
        im = ax.imshow(phi_grid, cmap='viridis', vmin=0, vmax=phi_max,
                        origin='lower', aspect='equal')
        # Annotate each cell with the phase value
        for i in range(N):
            for j in range(N):
                val = phi_grid[i, j]
                text_color = 'white' if val < phi_max * 0.6 else 'black'
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                        fontsize=7, color=text_color, fontweight='bold')

        ax.set_xticks(range(N))
        ax.set_xticklabels(z_labels, fontsize=7)
        ax.set_yticks(range(N))
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_xlabel("z (out-of-plane)  [cm]")
        ax.set_ylabel("y (in-plane)  [cm]")
        ax.set_title(label)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"Phase delay $\varphi_{ij}$  [rad]", fontsize=9)
        cb.ax.tick_params(labelsize=7)

    out = "phase_map.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved -> {out}")
    if show: plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Beam profiles (focused)
# ══════════════════════════════════════════════════════════════════════════════

def plot_beam_profiles(data, show=False):
    x_cm = cm(data['x_vec'])
    y_cm = cm(data['y_vec'])
    xf_cm = cm(float(data['x_focal']))
    yf_cm = cm(float(data['y_focal']))
    ix_f = int(data['ix_f'])
    iy_f = int(data['iy_f'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle("Beam Profiles through the Focal Point (Focused)",
                 fontsize=12, fontweight='bold')

    # ── Axial profile (y = y_focal) → slice along x ──
    ax = axes[0]
    # k-Wave field shape is [Nx, Ny], so axial cut at iy_f is [:, iy_f]
    ax.plot(x_cm, data['spl_arr1_foc'][:, iy_f],  color=ARRAY1_CLR, lw=1.2,
            ls='--', label="Array 1")
    ax.plot(x_cm, data['spl_arr2_foc'][:, iy_f],  color=ARRAY2_CLR, lw=1.2,
            ls=':',  label="Array 2")
    ax.plot(x_cm, data['spl_comb_foc'][:, iy_f],  color=COMB_CLR, lw=1.8,
            label="Combined")
    # Also show unfocused combined for reference
    ax.plot(x_cm, data['spl_comb_unfoc'][:, iy_f], color='#888888', lw=1.0,
            ls='-.', alpha=0.7, label="Unfocused (ref)")

    peak_c = float(data['spl_comb_foc'][ix_f, iy_f])
    ax.axhline(peak_c - 3, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(peak_c - 6, color='grey', lw=0.6, ls=':',  alpha=0.6)
    ax.text(x_cm[0]+0.3, peak_c-3+0.3, "-3 dB", fontsize=7, color='grey')
    ax.text(x_cm[0]+0.3, peak_c-6+0.3, "-6 dB", fontsize=7, color='grey')
    ax.axvline(xf_cm, color=FOCAL_CLR, lw=0.8, ls='--', alpha=0.7,
               label="focal x")

    ax.set_xlabel("x  [cm]")
    ax.set_ylabel("SPL  [dB re 20 $\\mu$Pa]")
    ax.set_title(f"Axial profile  (y = {yf_cm:.0f} cm)")
    ax.legend(fontsize=8, loc='lower right')

    # ── Lateral profile (x = x_focal) → slice along y ──
    ax = axes[1]
    ax.plot(data['spl_arr1_foc'][ix_f, :],  y_cm, color=ARRAY1_CLR, lw=1.2,
            ls='--', label="Array 1")
    ax.plot(data['spl_arr2_foc'][ix_f, :],  y_cm, color=ARRAY2_CLR, lw=1.2,
            ls=':',  label="Array 2")
    ax.plot(data['spl_comb_foc'][ix_f, :],  y_cm, color=COMB_CLR, lw=1.8,
            label="Combined")
    ax.plot(data['spl_comb_unfoc'][ix_f, :], y_cm, color='#888888', lw=1.0,
            ls='-.', alpha=0.7, label="Unfocused (ref)")

    peak_c_lat = float(data['spl_comb_foc'][ix_f, iy_f])
    ax.axvline(peak_c_lat - 3, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.axvline(peak_c_lat - 6, color='grey', lw=0.6, ls=':',  alpha=0.6)
    ax.axhline(yf_cm, color=FOCAL_CLR, lw=0.8, ls='--', alpha=0.7,
               label="focal y")

    ax.set_xlabel("SPL  [dB re 20 $\\mu$Pa]")
    ax.set_ylabel("y  [cm]")
    ax.set_title(f"Lateral profile  (x = {xf_cm:.0f} cm)")
    ax.legend(fontsize=8, loc='lower right')

    # Dark styling
    for a in axes:
        a.set_facecolor('#111111')
        a.grid(True, lw=0.3, alpha=0.4)
        a.spines[:].set_color('#444444')
        a.tick_params(colors='#CCCCCC')
        a.xaxis.label.set_color('#CCCCCC')
        a.yaxis.label.set_color('#CCCCCC')
        a.title.set_color('#EEEEEE')

    fig.patch.set_facecolor('#1A1A1A')
    out = "beam_profiles.png"
    plt.savefig(out, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved -> {out}")
    if show: plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualise simulation results")
    parser.add_argument("--input", default="results.npz",
                        help="path to results.npz produced by simulation.py")
    parser.add_argument("--show", action="store_true",
                        help="display figures interactively")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw = np.load(args.input, allow_pickle=True)
    data = dict(raw)

    # Unwrap 0-d arrays
    for key in data:
        if isinstance(data[key], np.ndarray) and data[key].ndim == 0:
            data[key] = data[key].item()

    # Recompute focal-point grid indices
    data['ix_f'] = int(np.argmin(np.abs(data['x_vec'] - data['x_focal'])))
    data['iy_f'] = int(np.argmin(np.abs(data['y_vec'] - data['y_focal'])))

    print("Generating figures ...")
    plt.style.use('dark_background')

    plot_unfocused_fields(data, show=args.show)
    plot_focused_fields(data, show=args.show)
    plot_focusing_comparison(data, show=args.show)
    plot_phase_map(data, show=args.show)
    plot_beam_profiles(data, show=args.show)

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
