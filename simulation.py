#!/usr/bin/env python3
"""
Ultrasonic Phased Array Beam Simulation — k-Wave Time-Domain
=============================================================
Two N-element linear arrays at 40 kHz, solved with the k-space
pseudospectral method (k-Wave-python, pure-Python backend).

The simulation runs in two stages:
  1. **Unfocused** — all element phases set to zero (natural radiation pattern)
  2. **Focused**   — delay-and-sum beamsteering to a common focal point

This demonstrates the physical effect of beamsteering: the unfocused arrays
radiate broad wavefronts that overlap incoherently, while the focused arrays
produce a tight constructive-interference peak at the focal point.

Run
---
    python simulation.py              # default N=8, focal at (25, 0) cm
    python simulation.py --N 12       # larger array
    python simulation.py --help       # all options
"""

import argparse
import time
import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

F0    = 40_000.0        # [Hz]     Operating frequency
C0    = 343.0           # [m/s]    Speed of sound in air at 20 °C
RHO0  = 1.204           # [kg/m³]  Air density at 20 °C, 1 atm
P_REF = 20e-6           # [Pa]     Reference pressure (SPL 0 dB threshold)

# Derived
OMEGA = 2.0 * np.pi * F0
K0    = OMEGA / C0                 # 732.73 rad/m
LAM   = C0 / F0                   # 8.575 mm


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ARRAY GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def make_array_geometry(N, array_sep, d_factor=0.5):
    """
    Return element positions for two N-element linear arrays on the y-axis.

    Both arrays sit at x = 0, separated vertically by `array_sep` metres.
    Element pitch is d_factor * λ (default λ/2, the Nyquist criterion).

    Returns
    -------
    x1, y1 : element positions of array 1 (upper) [m]
    x2, y2 : element positions of array 2 (lower) [m]
    d      : element pitch [m]
    """
    d = LAM * d_factor                        # default: λ/2 pitch
    off = (np.arange(N) - (N - 1) / 2) * d   # offsets from array centre

    y_c1 =  array_sep / 2                     # upper array centre
    y_c2 = -array_sep / 2                     # lower array centre

    x1 = np.full(N, 0.0);  y1 = y_c1 + off
    x2 = np.full(N, 0.0);  y2 = y_c2 + off

    return x1, y1, x2, y2, d, y_c1, y_c2


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FOCUSING PHASE LAW — delay-and-sum
# ══════════════════════════════════════════════════════════════════════════════

def focusing_phases(xe, ye, xf, yf, k):
    """
    Compute per-element phase advances for focused beamsteering.

    For constructive interference at the focal point, all wavefronts must
    arrive in phase.  A CW point source with phase φᵢ produces pressure:

        pᵢ ∝ sin(ωt − k·rᵢ + φᵢ)

    at the focal point.  For coherent summation we need:

        −k·rᵢ + φᵢ = const   for all i

    So  φᵢ = k·rᵢ + const.  Setting const = −k·rₘᵢₙ gives:

        φᵢ = k · (rᵢ − rₘᵢₙ)

    Elements farther from the focus get more phase advance (fire earlier)
    so their wavefronts arrive in sync with closer elements.

    Returns
    -------
    phi : phase advance per element [rad]  (all ≥ 0)
    r   : distance from each element to focal point [m]
    """
    r = np.sqrt((xf - xe)**2 + (yf - ye)**2)
    phi = k * (r - r.min())
    return phi, r


# ══════════════════════════════════════════════════════════════════════════════
# 4.  k-WAVE SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_kwave_sim(kgrid, medium, x_elem, y_elem, phases, amplitude,
                  pml_size, record_start_frac=0.5, quiet=False):
    """
    Run a single k-Wave 2D simulation with CW point sources.

    Each transducer element is a point source emitting:

        p_i(t) = A · sin(2π f₀ t + φᵢ)

    Parameters
    ----------
    kgrid   : kWaveGrid object (grid + time array already set)
    medium  : kWaveMedium object
    x_elem  : element x-positions in grid indices
    y_elem  : element y-positions in grid indices
    phases  : per-element phase [rad]
    amplitude : source amplitude [Pa]
    pml_size : PML thickness in grid points
    record_start_frac : fraction of simulation time before recording starts
                        (skip transient startup, capture only steady state)

    Returns
    -------
    p_max : 2D array of maximum pressure recorded at every grid point
            during the steady-state portion of the simulation
    """
    n_elem = len(x_elem)
    Nt = kgrid.Nt
    dt = kgrid.dt

    # Build source
    source = kSource()
    source.p_mask = np.zeros([kgrid.Nx, kgrid.Ny], dtype=bool)

    # Map element positions to grid indices — sources are placed on the mask
    for ix, iy in zip(x_elem, y_elem):
        source.p_mask[ix, iy] = True

    # k-Wave assigns source.p rows to mask True-positions in raster order
    # (linear index = ix * Ny + iy).  We must sort the element signals to
    # match that ordering, otherwise phases get applied to wrong elements.
    linear_idx = np.asarray(x_elem) * kgrid.Ny + np.asarray(y_elem)
    sort_order = np.argsort(linear_idx)

    # Time-domain CW signals with per-element phase (sorted to match mask)
    t_arr = np.arange(Nt) * dt
    signals = np.zeros((n_elem, Nt))
    for row, i in enumerate(sort_order):
        signals[row, :] = amplitude * np.sin(2 * np.pi * F0 * t_arr + phases[i])

    source.p = signals
    source.p_mode = 'additive'

    # Sensor: record p_max everywhere, but only during steady state
    sensor = kSensor()
    sensor.mask = np.ones([kgrid.Nx, kgrid.Ny], dtype=bool)
    sensor.record = ['p_max']
    sensor.record_start_index = int(Nt * record_start_frac)

    result = kspaceFirstOrder(
        kgrid=kgrid,
        source=source,
        medium=medium,
        sensor=sensor,
        backend='python',
        pml_inside=True,
        pml_size=pml_size,
        quiet=quiet,
    )

    p_max = result['p_max'].reshape(kgrid.Nx, kgrid.Ny)
    return p_max


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SPL CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def to_spl(p_peak, p_ref=P_REF):
    """Peak pressure → SPL [dB re 20 μPa].  p_rms = p_peak / √2."""
    p_rms = np.abs(p_peak) / np.sqrt(2.0)
    return 20.0 * np.log10(np.maximum(p_rms, p_ref * 1e-4) / p_ref)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN SIMULATION ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(N=8, array_sep=0.30, x_focal=0.25, y_focal=0.00,
                   A_elem=10.0, d_factor=0.5, ppw=6, n_periods=55,
                   pml_size=20, verbose=True, progress_cb=None,
                   run_individual=True):
    """
    Full simulation pipeline:
      1. Build grid and place arrays
      2. Run UNFOCUSED simulation (φ = 0 for all elements)
      3. Compute focusing phases (delay-and-sum)
      4. Run FOCUSED simulation
      5. Return all results

    Parameters
    ----------
    N         : elements per array (the N-element 2D cross-section of N×N)
    array_sep : vertical separation between array centres [m]
    x_focal   : focal point x [m]
    y_focal   : focal point y [m]
    A_elem    : source amplitude per element [Pa]
    d_factor  : element spacing as multiple of λ (0.5 = λ/2 Nyquist)
    ppw       : points per wavelength (k-Wave recommends ≥ 6)
    n_periods : number of CW periods to simulate (need ~10 for steady state)
    pml_size  : PML layer thickness [grid points]
    progress_cb : callable(step, total, description) for progress updates
    run_individual : if False, skip individual-array sims (faster)
    """

    # ── Grid setup ────────────────────────────────────────────────────────
    dx = LAM / ppw    # grid spacing [m]

    # Physical domain extent
    x_min, x_max = -0.05, x_focal + 0.10
    y_min, y_max = -(array_sep / 2 + 0.10), +(array_sep / 2 + 0.10)

    Nx = int(np.ceil((x_max - x_min) / dx))
    Ny = int(np.ceil((y_max - y_min) / dx))

    # k-Wave grid (note: k-Wave uses [Nx, Ny] ordering)
    kgrid = kWaveGrid([Nx, Ny], [dx, dx])

    # Physical coordinate vectors (centred at grid midpoint by k-Wave)
    # k-Wave centres the grid at (0,0), so we need to map our physical
    # coordinates into k-Wave's coordinate system.
    #
    # k-Wave x range: [-Nx*dx/2, Nx*dx/2]
    # Our physical x range: [x_min, x_max]
    # Offset: our_x = kwave_x + x_offset
    x_offset = (x_min + x_max) / 2
    y_offset = (y_min + y_max) / 2

    # Coordinate vectors in physical space
    x_vec = np.linspace(-Nx * dx / 2, Nx * dx / 2, Nx) + x_offset
    y_vec = np.linspace(-Ny * dx / 2, Ny * dx / 2, Ny) + y_offset

    if verbose:
        sep = "═" * 60
        print(sep)
        print("  ULTRASONIC PHASED ARRAY — k-WAVE TIME-DOMAIN SIMULATION")
        print(sep)
        print(f"  Frequency       f₀  = {F0/1e3:.0f} kHz")
        print(f"  Wavelength      λ   = {LAM*1e3:.3f} mm")
        print(f"  Grid spacing    dx  = {dx*1e3:.3f} mm  (λ/{ppw})")
        print(f"  Grid size            = {Nx} × {Ny} = {Nx*Ny:,} points")
        print(f"  Physical domain      = [{x_min*100:.0f}, {x_max*100:.0f}] × "
              f"[{y_min*100:.0f}, {y_max*100:.0f}] cm")
        print(f"  PML size             = {pml_size} points")

    # ── Array geometry ────────────────────────────────────────────────────
    x1, y1, x2, y2, d, y_c1, y_c2 = make_array_geometry(N, array_sep, d_factor)

    if verbose:
        aper = (N - 1) * d
        z_R = aper**2 / (4 * LAM)
        print(f"  Elements/array  N   = {N}")
        print(f"  Element pitch   d   = {d*1e3:.3f} mm  (λ/2)")
        print(f"  Aperture        D   = {aper*1e3:.1f} mm")
        print(f"  Rayleigh dist   z_R = {z_R*1e2:.2f} cm")
        print(f"  Array separation    = {array_sep*100:.0f} cm")
        print(f"  Focal point         = ({x_focal*100:.0f}, {y_focal*100:.0f}) cm")
        print(sep)

    # Map element positions to k-Wave grid indices
    def phys_to_idx(px, py):
        """Convert physical (x, y) [m] to k-Wave grid indices."""
        ix = np.round((px - x_offset) / dx + (Nx - 1) / 2).astype(int)
        iy = np.round((py - y_offset) / dx + (Ny - 1) / 2).astype(int)
        return ix, iy

    ix1, iy1 = phys_to_idx(x1, y1)
    ix2, iy2 = phys_to_idx(x2, y2)

    # Verify mapping
    if verbose:
        print(f"\n  Array 1 element grid indices (ix): {ix1}")
        print(f"  Array 1 element grid indices (iy): {iy1}")
        print(f"  Array 2 element grid indices (ix): {ix2}")
        print(f"  Array 2 element grid indices (iy): {iy2}")

    # ── Time stepping ─────────────────────────────────────────────────────
    t_end = n_periods / F0
    kgrid.makeTime(C0, cfl=0.3, t_end=t_end)

    if verbose:
        print(f"\n  CW periods           = {n_periods}")
        print(f"  Time steps           = {kgrid.Nt}")
        print(f"  dt                   = {kgrid.dt*1e6:.3f} μs")
        print(f"  Simulation time      = {t_end*1e3:.3f} ms")
        print()

    # ── Medium ────────────────────────────────────────────────────────────
    medium = kWaveMedium(sound_speed=C0, density=RHO0)

    # ── Focusing phases ───────────────────────────────────────────────────
    phi1_focus, r1 = focusing_phases(x1, y1, x_focal, y_focal, K0)
    phi2_focus, r2 = focusing_phases(x2, y2, x_focal, y_focal, K0)

    if verbose:
        print("  Focusing phases (delay-and-sum):")
        print(f"    Array 1 φ [rad]: {np.round(phi1_focus, 3)}")
        print(f"    Array 2 φ [rad]: {np.round(phi2_focus, 3)}")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: UNFOCUSED SIMULATION (all phases = 0)
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 60)
        print("  STAGE 1: UNFOCUSED (all φ = 0)")
        print("━" * 60)

    phi_zero = np.zeros(N)

    # Combine both arrays into one simulation for efficiency
    ix_all = np.concatenate([ix1, ix2])
    iy_all = np.concatenate([iy1, iy2])
    phi_unfocused = np.concatenate([phi_zero, phi_zero])

    total_sims = 6 if run_individual else 2
    sim_step = 0

    def _progress(desc):
        nonlocal sim_step
        sim_step += 1
        if progress_cb:
            progress_cb(sim_step, total_sims, desc)

    p_max_arr1_unfoc = None
    p_max_arr2_unfoc = None
    p_max_arr1_foc = None
    p_max_arr2_foc = None

    if run_individual:
        # Array 1 alone (unfocused)
        if verbose:
            print("\n  Running Array 1 (unfocused)...")
        t0 = time.time()
        p_max_arr1_unfoc = run_kwave_sim(
            kgrid, medium, ix1, iy1, phi_zero, A_elem,
            pml_size, record_start_frac=0.5, quiet=not verbose,
        )
        t1 = time.time()
        if verbose:
            print(f"    Done in {t1-t0:.1f} s")
        _progress("Array 1 unfocused done")

        # Array 2 alone (unfocused)
        if verbose:
            print("  Running Array 2 (unfocused)...")
        p_max_arr2_unfoc = run_kwave_sim(
            kgrid, medium, ix2, iy2, phi_zero, A_elem,
            pml_size, record_start_frac=0.5, quiet=not verbose,
        )
        t2 = time.time()
        if verbose:
            print(f"    Done in {t2-t1:.1f} s")
        _progress("Array 2 unfocused done")
    else:
        t0 = time.time()
        t2 = t0

    # Both arrays combined (unfocused)
    if verbose:
        print("  Running Combined (unfocused)...")
    p_max_comb_unfoc = run_kwave_sim(
        kgrid, medium, ix_all, iy_all, phi_unfocused, A_elem,
        pml_size, record_start_frac=0.5, quiet=not verbose,
    )
    t3 = time.time()
    if verbose:
        print(f"    Done in {t3-t2:.1f} s")
    _progress("Combined unfocused done")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: FOCUSED SIMULATION (delay-and-sum beamsteering)
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "━" * 60)
        print("  STAGE 2: FOCUSED (delay-and-sum beamsteering)")
        print("━" * 60)

    phi_focused = np.concatenate([phi1_focus, phi2_focus])

    if run_individual:
        # Array 1 alone (focused)
        if verbose:
            print("\n  Running Array 1 (focused)...")
        p_max_arr1_foc = run_kwave_sim(
            kgrid, medium, ix1, iy1, phi1_focus, A_elem,
            pml_size, record_start_frac=0.5, quiet=not verbose,
        )
        t4 = time.time()
        if verbose:
            print(f"    Done in {t4-t3:.1f} s")
        _progress("Array 1 focused done")

        # Array 2 alone (focused)
        if verbose:
            print("  Running Array 2 (focused)...")
        p_max_arr2_foc = run_kwave_sim(
            kgrid, medium, ix2, iy2, phi2_focus, A_elem,
            pml_size, record_start_frac=0.5, quiet=not verbose,
        )
        t5 = time.time()
        if verbose:
            print(f"    Done in {t5-t4:.1f} s")
        _progress("Array 2 focused done")
    else:
        t5 = t3

    # Both arrays combined (focused)
    if verbose:
        print("  Running Combined (focused)...")
    p_max_comb_foc = run_kwave_sim(
        kgrid, medium, ix_all, iy_all, phi_focused, A_elem,
        pml_size, record_start_frac=0.5, quiet=not verbose,
    )
    t6 = time.time()
    if verbose:
        print(f"    Done in {t6-t5:.1f} s")
    _progress("Combined focused done")

    # ── Focal point grid index ────────────────────────────────────────────
    ix_f = np.argmin(np.abs(x_vec - x_focal))
    iy_f = np.argmin(np.abs(y_vec - y_focal))

    # ── Convert to SPL ────────────────────────────────────────────────────
    spl_arr1_unfoc = to_spl(p_max_arr1_unfoc) if p_max_arr1_unfoc is not None else None
    spl_arr2_unfoc = to_spl(p_max_arr2_unfoc) if p_max_arr2_unfoc is not None else None
    spl_comb_unfoc = to_spl(p_max_comb_unfoc)
    spl_arr1_foc = to_spl(p_max_arr1_foc) if p_max_arr1_foc is not None else None
    spl_arr2_foc = to_spl(p_max_arr2_foc) if p_max_arr2_foc is not None else None
    spl_comb_foc = to_spl(p_max_comb_foc)

    # ── Print summary ─────────────────────────────────────────────────────
    if verbose:
        print("\n" + "═" * 60)
        print("  RESULTS AT FOCAL POINT")
        print("═" * 60)
        print(f"\n  UNFOCUSED:")
        if spl_arr1_unfoc is not None:
            print(f"    SPL Array 1:   {spl_arr1_unfoc[ix_f, iy_f]:.1f} dB")
            print(f"    SPL Array 2:   {spl_arr2_unfoc[ix_f, iy_f]:.1f} dB")
        print(f"    SPL Combined:  {spl_comb_unfoc[ix_f, iy_f]:.1f} dB")
        print(f"    Peak |p|:      {p_max_comb_unfoc[ix_f, iy_f]:.3f} Pa")
        print(f"\n  FOCUSED (beamsteered):")
        if spl_arr1_foc is not None:
            print(f"    SPL Array 1:   {spl_arr1_foc[ix_f, iy_f]:.1f} dB")
            print(f"    SPL Array 2:   {spl_arr2_foc[ix_f, iy_f]:.1f} dB")
        print(f"    SPL Combined:  {spl_comb_foc[ix_f, iy_f]:.1f} dB")
        print(f"    Peak |p|:      {p_max_comb_foc[ix_f, iy_f]:.3f} Pa")
        gain = spl_comb_foc[ix_f, iy_f] - spl_comb_unfoc[ix_f, iy_f]
        print(f"\n    Focusing gain: {gain:+.1f} dB")
        print(f"\n  Total wall time: {t6-t0:.1f} s")
        print()

    return dict(
        # Grid
        x_vec=x_vec, y_vec=y_vec, dx=dx, Nx=Nx, Ny=Ny,
        x_offset=x_offset, y_offset=y_offset,
        # Config
        N=N, array_sep=array_sep, d=d, d_factor=d_factor, ppw=ppw,
        A_elem=A_elem, x_focal=x_focal, y_focal=y_focal,
        ix_f=ix_f, iy_f=iy_f,
        x1=x1, y1=y1, x2=x2, y2=y2, y_c1=y_c1, y_c2=y_c2,
        # Phases
        phi1_focus=phi1_focus, phi2_focus=phi2_focus, r1=r1, r2=r2,
        # Unfocused fields (p_max)
        p_max_arr1_unfoc=p_max_arr1_unfoc,
        p_max_arr2_unfoc=p_max_arr2_unfoc,
        p_max_comb_unfoc=p_max_comb_unfoc,
        spl_arr1_unfoc=spl_arr1_unfoc,
        spl_arr2_unfoc=spl_arr2_unfoc,
        spl_comb_unfoc=spl_comb_unfoc,
        # Focused fields (p_max)
        p_max_arr1_foc=p_max_arr1_foc,
        p_max_arr2_foc=p_max_arr2_foc,
        p_max_comb_foc=p_max_comb_foc,
        spl_arr1_foc=spl_arr1_foc,
        spl_arr2_foc=spl_arr2_foc,
        spl_comb_foc=spl_comb_foc,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ultrasonic phased array simulation (k-Wave time-domain)")
    parser.add_argument("--N",      type=int,   default=8,    help="elements per array")
    parser.add_argument("--sep",    type=float, default=0.30, help="array separation [m]")
    parser.add_argument("--xf",     type=float, default=0.25, help="focal x [m]")
    parser.add_argument("--yf",     type=float, default=0.00, help="focal y [m]")
    parser.add_argument("--A",      type=float, default=10.0, help="element amplitude [Pa]")
    parser.add_argument("--ppw",    type=int,   default=6,    help="points per wavelength")
    parser.add_argument("--periods",type=int,   default=55,   help="CW periods to simulate")
    parser.add_argument("--pml",    type=int,   default=20,   help="PML size [grid points]")
    parser.add_argument("--save",   type=str,   default="results.npz",
                        help="output filename")
    args = parser.parse_args()

    results = run_simulation(
        N=args.N, array_sep=args.sep,
        x_focal=args.xf, y_focal=args.yf,
        A_elem=args.A, ppw=args.ppw,
        n_periods=args.periods, pml_size=args.pml,
    )

    # Save all numpy arrays
    save_dict = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            save_dict[k] = v

    np.savez_compressed(args.save, **save_dict)
    print(f"Results saved → {args.save}")
