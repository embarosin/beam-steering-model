# Ultrasonic Phased Array Beam Simulation

Two N-element linear arrays at 40 kHz beamsteered to a common focal point,
solved with the **k-Wave pseudospectral time-domain** method.

The simulation runs in two stages:
1. **Unfocused** — all element phases set to zero (natural radiation pattern)
2. **Focused** — delay-and-sum beamsteering to a common focal point

This demonstrates the physical effect of beamsteering: the unfocused arrays
radiate broad wavefronts that overlap incoherently, while the focused arrays
produce a tight constructive-interference peak at the focal point (+18 dB gain).

---

## Setup

### Prerequisites

- Python >= 3.10
- pip (comes with Python)
- A web browser (for the interactive tool)

### Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd beam-steer-model
pip install -r requirements.txt
```

This installs numpy, scipy, matplotlib, k-Wave-python, and gradio.

---

## Run

### Command-line simulation

```bash
# Full simulation + save results.npz  (~2 min)
python simulation.py

# Generate all figures from saved results
python visualizer.py

# Larger array
python simulation.py --N 16

# Move focal point
python simulation.py --xf 0.20 --yf 0.05

# All options
python simulation.py --help
```

### Interactive tool

```bash
python interactive.py
```

This launches a web-based GUI at **http://localhost:7860** where you can adjust
array parameters (number of elements, spacing, separation, source amplitude,
focal point position), run the simulation, and visualise the results with a
progress bar.

**If the browser doesn't open automatically (common on macOS):**

```bash
# Option 1: open manually
open http://localhost:7860

# Option 2: run with explicit browser launch
python -c "import webbrowser; webbrowser.open('http://localhost:7860')" &
python interactive.py
```

You can also create a shareable public link:

```bash
python interactive.py --share
```

---

## Output files

### From `simulation.py` + `visualizer.py`

| File | Description |
|---|---|
| `results.npz` | All field arrays (load with `np.load`) |
| `unfocused_fields.png` | Array 1 / Array 2 / Combined SPL — no beamsteering |
| `focused_fields.png` | Array 1 / Array 2 / Combined SPL — beamsteered |
| `focusing_comparison.png` | Side-by-side unfocused vs. focused combined field |
| `phase_map.png` | Per-element phase delays for each array |
| `beam_profiles.png` | Axial and lateral cuts through focal point |

### Written deliverables

| File | Description |
|---|---|
| `REPORT.md` | Technical report covering the physics, solver, and results |
| `assumptions_log.md` | Every physical assumption with one-line justification |
| `interaction_model.md` | Beam interaction model write-up |
| `next_steps.md` | What I'd do with another two hours |

---

## Solver: k-Wave Time-Domain

This simulation uses [k-Wave-python](https://github.com/waltsims/k-wave-python)
(v0.6.1), the Python port of the k-Wave MATLAB toolbox. The solver implements a
**k-space pseudospectral method** — a time-domain FDTD-like scheme that computes
spatial derivatives in the Fourier domain, giving spectral accuracy on a
relatively coarse grid.

**Why k-Wave for this problem:**

- Industry standard for computational acoustics — widely validated
- Full time-domain physics — captures transient startup, reflections, and
  nonlinear propagation naturally
- Pseudospectral accuracy — PPW >= 6 is sufficient (vs. PPW >= 10 for
  standard FDTD) due to k-space correction
- PML absorbing boundaries — prevents artificial reflections from domain edges

**Simulation parameters:**

| Parameter | Value | Notes |
|---|---|---|
| Grid spacing dx | lambda/6 = 1.43 mm | 6 points per wavelength (pseudospectral accuracy) |
| CFL number | 0.3 | Ensures numerical stability |
| CW periods | 55 | Enough for propagation to focal point + steady state |
| PML thickness | 20 grid points | Absorbing boundary layer |
| Record start | 50% of simulation | Discard transient startup |

---

## Physics

### Source Model

Each transducer element is a **CW point source** emitting:

```
p_i(t) = A * sin(2*pi*f0*t + phi_i)
```

where A = 10 Pa (source amplitude) and phi_i is the focusing phase delay.
Sources are driven in `additive` mode in k-Wave, meaning the source pressure
is added to the existing field at each time step.

### Phase Law — Delay-and-Sum Focusing

A CW point source with phase phi_i produces pressure at distance r_i:

```
p_i(r_i, t) ~ sin(omega*t - k*r_i + phi_i)
```

For all wavefronts to arrive in-phase at the focal point, we need
`-k*r_i + phi_i = constant` for all i. Solving:

```
phi_i = k * (r_i - r_min),   r_i = |r_elem_i - r_focal|
```

`r_min = min(r_i)` sets the zero-phase reference at the element closest to
the focus. Farther elements receive more phase advance (fire earlier in the
cycle), compensating for their longer travel time.

**Unfocused case:** All phi_i = 0 — elements emit in phase, producing a
broadside radiation pattern.

**Focused case:** Phase advances computed from the delay-and-sum law above,
steering the beam toward the common focal point.

**Example (Array 1, N=8, focal point at (25, 0) cm):**

| Element y [cm] | Distance r [cm] | Phase phi [rad] |
|---|---|---|
| 13.5 (closest) | 28.41 | 0.00 |
| 14.8 | 28.76 | 2.57 |
| 15.6 | 29.31 | 6.60 |
| 16.5 (farthest) | 29.95 | 11.30 |

### Beam Interaction Model

**Primary interaction: linear superposition**

Both arrays are run simultaneously in a single k-Wave simulation. The solver
naturally computes the full wave interaction — the pressure fields from both
arrays propagate, interfere, and superpose exactly as they would physically.
This is more accurate than post-hoc addition of separate fields because the
time-domain solver captures:

- Constructive/destructive interference at every grid point
- The correct temporal evolution of the combined field
- Boundary reflections (absorbed by PML) from the combined field

At the focal point, where both beams arrive in-phase by construction, the
combined field is +18.3 dB above the unfocused baseline.

**What linear superposition misses:**

The most physically significant effect omitted by pure superposition is
**acoustic streaming**. At the intersection of the two beams, the time-averaged
momentum flux of the acoustic field drives a steady DC flow in the direction of
beam propagation (Eckart streaming). This flow can be substantial — estimates
for focused 40 kHz fields in air give streaming velocities of O(cm/s) — and it
would:

1. Advect the wavefronts from both arrays, slightly Doppler-shifting the
   apparent frequency at the focal region.
2. Create a convective acoustic non-reciprocity: waves propagating into the
   stream vs. against it accumulate different phases.
3. Modify the local effective sound speed, subtly detuning the constructive
   interference condition at the focal point.

**Parametric array interaction** — at the intersection, the sum- and
difference-frequency waves generated by nonlinear mixing of the two beams
create a secondary source that radiates collinearly with the bisector of the
two beams (Berktay parametric source mechanism). k-Wave can capture this
natively by enabling the `BonA` nonlinearity parameter in the medium, which
would be a natural next step.

---

## Assumptions Log

| Assumption | Value | Justification |
|---|---|---|
| Speed of sound | 343 m/s | Air at 20 C, 1 atm (ISO 9613) |
| Air density | 1.204 kg/m3 | Standard atmosphere at 20 C |
| Element spacing d | lambda/2 = 4.287 mm | Nyquist criterion: suppresses grating lobes |
| Elements per array N | 8 | Realistic for a compact hand-held array |
| Array separation | 30 cm | Provides ~30 degree steering angle to focal point |
| Focal point | (25, 0) cm | Far-field focus for both arrays |
| Element amplitude A | 10 Pa | Moderate piezo drive level |
| 2D model | yes | N*N array -> N-element linear array in the propagation plane |
| Lossless medium | yes | Absorption at 40 kHz over 30 cm is < 0.3 dB (ISO 9613) |
| CW steady state | yes | Record only after 50% of simulation time |
| Linear medium | yes | Peak pressure ~4.4 Pa << rho*c^2 ~ 142 kPa |
| Grid resolution | lambda/6 (PPW=6) | Sufficient for k-space pseudospectral method |
| PML boundaries | 20 points | Absorbs outgoing waves, prevents reflections |
| CFL = 0.3 | Stable | Well below the stability limit for k-space methods |

---

## What I'd Do Next

With another two hours I would push in three directions. First, I'd enable
nonlinear propagation by setting `medium.BonA = 0.4` (the B/A parameter for
air) — k-Wave natively supports the Westervelt equation in its time-domain
solver, so this would capture second-harmonic generation, sum/difference
frequency mixing, and parametric array effects at the beam intersection
without any additional code beyond setting the medium property. Second, I'd
add thermoviscous absorption via `medium.alpha_coeff` and
`medium.alpha_power` — at 40 kHz the absorption is ~1 dB/m in air, modest over
our 30 cm path but significant at the 80 kHz second harmonic (~4 dB/m), and
k-Wave's frequency-dependent absorption model handles this naturally. Third,
I'd run the full 8x8 array in a 3D simulation to capture focusing in both
transverse dimensions — the focusing gain should roughly double (in dB)
compared to the 2D case, though the grid would be approximately 350^3 points
instead of 280 x 350, so the compute cost is substantially higher.
