# Technical Report: Ultrasonic Phased Array Beam Simulation

## Overview

This simulation models two ultrasonic transducer arrays operating at 40 kHz
in air. Each array is an 8x8 grid of piezoelectric elements. The two arrays
are positioned on the same vertical line (same x-position), separated by
30 cm, and both are steered to focus at a common point at (25, 0) cm.

The question being answered: given the right phase delay on each element, can
we get the two beams to constructively interfere at a chosen point in space,
and how much pressure gain results? The simulation shows +18.3 dB of focusing
gain at the target.

---

## Physical setup

```
                    y [cm]
                     ^
                     |
    Array 1 ------> |  o o o o o o o o     (8 elements at y = +13.5 to +16.5 cm)
                     |
                     |                              * focal point (25, 0) cm
    - - - - - - - - -+- - - - - - - - - - - - - - - -> x [cm]
                     |
    Array 2 ------> |  o o o o o o o o     (8 elements at y = -16.5 to -13.5 cm)
                     |
```

Both arrays sit at x = 0. In the full 3D problem each array is 8x8, but since
we're simulating in 2D (the x-y propagation plane), we model the cross-section:
one row of 8 elements per array. The out-of-plane dimension (z) is treated as
infinite and uniform -- this is a standard 2D acoustic approximation.

The NxN array has symmetry in the z-direction for a focal point at z = 0, so
the 2D slice captures the relevant physics (steering angle, focusing gain,
interference pattern) without the cost of a 3D solve. The phase map figure
does show the full 8x8 pattern to illustrate what happens in the third
dimension.

---

## Physical parameters

| Parameter | Value | Rationale |
|---|---|---|
| Frequency f | 40 kHz | Specified by the problem. Lambda = 8.575 mm in air. |
| Speed of sound c | 343 m/s | Air at 20 C, 1 atm. |
| Element pitch d | lambda/2 = 4.29 mm | Spatial Nyquist criterion. Spacing wider than lambda/2 produces grating lobes -- parasitic beams at unwanted angles, analogous to aliasing in signal processing. |
| Array aperture D | 30.0 mm | (N-1) * d. Total span of 8 elements. |
| Array separation | 30 cm | Gives a steering angle of about 31 degrees from horizontal to the focal point. |
| Focal point | (25, 0) cm | Positioned on the y = 0 midline so both arrays steer symmetrically. |
| Source amplitude | 10 Pa per element | Peak combined pressure at focus reaches ~4.4 Pa, well within the linear acoustic regime (small compared to atmospheric pressure ~101 kPa). |

---

## Beamsteering: the phase law

This is the central piece of physics in the simulation. Each transducer
element emits a continuous-wave sinusoidal pressure signal:

```
p_i(t) = A * sin(omega * t + phi_i)
```

where phi_i is the phase assigned to element i. As the wave propagates outward,
at distance r_i from the element the pressure becomes:

```
p_i(r_i, t)  ~  (A / r_i) * sin(omega * t  -  k * r_i  +  phi_i)
```

The `-k * r_i` term is the propagation phase -- the wave accumulates phase as
it travels. Greater distance means more accumulated phase, so the wavefront
arrives later in the cycle.

The goal is to make all N elements' contributions arrive at the focal point
with the same phase so they sum constructively. At the focal point, element i's
contribution has total phase:

```
theta_i  =  omega * t  -  k * r_i  +  phi_i
```

For all elements to be in-phase, we need theta_i = theta_j for all pairs i, j:

```
-k * r_i + phi_i  =  -k * r_j + phi_j    (for all i, j)
```

Rearranging: phi_i = k * r_i + constant.

We choose the constant so the minimum phase is zero:

```
phi_i  =  k * (r_i  -  r_min)
```

The physical picture: elements farther from the focal point get more phase
advance -- they fire earlier in the sinusoidal cycle, which compensates for
their longer propagation path. The closest element (r_min) gets zero phase
and serves as the reference.

For Array 1 (upper), the closest element to (25, 0) cm is the bottom one at
y = 13.5 cm, with r_min = 28.41 cm. The farthest element (y = 16.5 cm,
r = 29.95 cm) gets phi = k * 1.54 cm = 11.3 rad. That's about 1.8 full
cycles of lead -- it fires almost two wavelengths earlier to make up for its
extra distance.

---

## Solver and simulation method

The simulation uses k-Wave, a computational acoustics solver that implements
the k-space pseudospectral method. The procedure:

1. Discretize the 2D domain into a grid (280 x 350 points, spacing = lambda/6).
2. Place sources on the grid. Each transducer element occupies one grid point
   and injects its sinusoidal signal at each time step.
3. Step the acoustic wave equation forward in time. At each step, k-Wave
   computes spatial pressure gradients via FFTs (the pseudospectral part, which
   is more accurate per grid point than finite differences), applies a k-space
   correction to reduce numerical dispersion, updates the velocity and pressure
   fields, and absorbs outgoing waves at the domain boundaries through a PML
   (Perfectly Matched Layer).
4. Record the maximum pressure at every grid point during the steady-state
   portion (the second half of the simulation, after transients have died out).

The simulation runs 6 times:
- Array 1 alone, Array 2 alone, and both combined -- each for the unfocused
  case (all phases zero) and the focused case (beamsteered).

This produces individual and combined field maps for both cases, making it
possible to see each array's contribution separately.

An alternative would be to sum the fields analytically using the Green's
function for a 2D point source. k-Wave solves the full wave equation instead,
which means diffraction, near-field effects, and wave interactions are handled
by the solver rather than approximated. The combined-field simulation also
captures the actual superposition as it evolves in time, rather than adding
pre-computed steady-state fields.

---

## Results

### Unfocused (all phases = 0)

Without beamsteering, each array radiates a broad wavefront perpendicular to
its element line (the broadside direction, +x). The two wavefronts spread and
overlap. At the intended focal point:

- Single array: 79.9 dB SPL
- Combined: 85.6 dB (about +6 dB over a single array)

The +6 dB is consistent with expectations: two coherent sources of equal
amplitude double the pressure, and 20 * log10(2) = 6.02 dB. The two arrays
are symmetric about y = 0, so even without focusing their contributions
arrive roughly in-phase at that point.

### Focused (delay-and-sum beamsteering)

With the computed phase delays:

- Single array: 98.2 dB SPL at the focal point
- Combined: 103.9 dB
- Focusing gain: +18.3 dB over the unfocused combined field

The single-array focusing gain (~18 dB, from 79.9 to 98.2) comes from
redistributing the acoustic energy spatially. Instead of a broad wavefront,
the phased elements produce a converging wavefront that concentrates pressure
at the focal point.

The combined focused field is ~+6 dB above a single focused array
(103.9 vs 98.2), consistent with coherent addition of two equal-amplitude
sources.

### Beam profile

The axial profile (along y = 0, through the focal point) shows a peak at
x = 25 cm with a -3 dB width of about 2-3 cm, and deep nulls where sidelobes
destructively interfere.

The lateral profile (along x = 25 cm) shows the focused spot is narrower in y
than in x. This is because the effective aperture in the y-direction is the
full 30 cm array separation, which is much larger than the 3 cm element
aperture of a single array.

---

## Phase map

The phase map figure shows the full 8x8 element grid for each array, with the
phase value at each element displayed as a heatmap.

- The pattern is roughly concentric: elements equidistant from the focal point
  receive the same phase, producing the curved contours expected for a
  converging wavefront.
- The two arrays' phase maps are mirror images of each other (flipped in y),
  since the focal point lies on the symmetry axis.
- Corner elements get the largest phase values because they are farthest from
  the focal point in 3D and need the most compensation.
- The center column (z = 0) corresponds to the row of elements actually used
  in the 2D simulation. The full 8x8 map represents what would be programmed
  into the array's phase control electronics for a 3D system.

---

## Beam interaction model

### Linear superposition

The combined simulation runs both arrays simultaneously in a single k-Wave
solve. The wave equation is linear for small-amplitude acoustics, so the
total pressure at any point is the sum of the individual contributions. k-Wave
computes this by propagating all sources together on the same grid, which
accounts for:

- Constructive interference at the focal point (by design of the phase law)
- Destructive interference elsewhere (sidelobes and nulls)
- Diffraction effects as the beams overlap

### What linear superposition does not capture

The main omission is nonlinear acoustic effects, which matter when pressure
amplitudes become non-negligible compared to the ambient pressure. Two
phenomena are relevant here:

**Harmonic generation and parametric mixing.** At the beam intersection, where
the combined amplitude is highest, the nonlinear term in the wave equation
(proportional to p^2) generates second harmonics at 80 kHz from each beam
individually, a sum-frequency component at 80 kHz from the cross-interaction,
and a difference-frequency component at 0 Hz (a DC radiation pressure). The
difference-frequency mechanism is the basis of the parametric array effect
(Berktay), where two ultrasonic beams mix to produce a directional
audible-frequency beam. At the pressure levels in this simulation (~4 Pa
peak), these effects would be present but small.

**Acoustic streaming.** Time-averaged momentum transfer from the acoustic
field to the medium drives a steady bulk flow (Eckart streaming). At the beam
intersection, this produces a localized flow of air in the beam propagation
direction. Streaming velocity scales with the square of the pressure amplitude
and inversely with viscosity. For focused 40 kHz fields in air, rough
estimates give flow speeds on the order of cm/s -- enough to advect
wavefronts slightly and modify the interference pattern.

k-Wave does support nonlinear propagation through the medium's B/A parameter.
Setting `medium.BonA = 0.4` (the value for air) would capture harmonic
generation and parametric mixing without further code changes.

---

## Assumptions

| # | Assumption | Justification |
|---|---|---|
| 1 | Air at 20 C, 1 atm (c = 343 m/s, rho = 1.204 kg/m^3) | Standard conditions. c varies with temperature (~0.6 m/s per degree C), but the exact value does not affect the conclusions here. |
| 2 | Lossless medium (no absorption) | At 40 kHz, atmospheric absorption is about 1.2 dB/m (ISO 9613-1). Over our 30 cm path that is ~0.36 dB, negligible next to the 18 dB focusing gain. At 80 kHz (second harmonic) absorption is ~4 dB/m, which would matter for nonlinear modeling. |
| 3 | Linear acoustics | Peak pressure ~4.4 Pa. Acoustic Mach number p/(rho*c^2) ~ 3e-5, well below the ~0.01 threshold for significant nonlinear effects. Superposition is a good approximation. |
| 4 | Point sources (no element directivity) | Real elements have a radiation pattern that rolls off at large angles. For elements small compared to lambda (4.3 mm pitch vs 8.6 mm wavelength), the pattern is close to omnidirectional. Reasonable for lambda/2-spaced elements. |
| 5 | 2D simulation (infinite in z) | The NxN array focuses in z as well, but for a focal point on the z = 0 plane, the 2D cross-section captures the steering and interference physics. The main omission is the additional focusing gain from the z-dimension. |
| 6 | Lambda/2 element spacing | Spatial Nyquist criterion. Wider spacing undersamples the wavefield and produces grating lobes. At exactly lambda/2, the first grating lobe is at 90 degrees (endfire), where it is suppressed. |
| 7 | CW steady state | 55 periods simulated (1.375 ms), recording the second half. The longest propagation path is ~30 cm, requiring ~0.87 ms (35 periods). Recording begins at period 28; waves reach the focal point by period 35. That leaves about 20 periods of steady-state data, which is sufficient for p_max to converge. |
| 8 | PPW = 6 (grid spacing = lambda/6) | The pseudospectral method has spectral accuracy in space, so 6 points per wavelength is adequate. A standard finite-difference scheme would need 10-20 PPW. |
| 9 | PML absorbing boundaries (20 grid points) | The Perfectly Matched Layer absorbs outgoing waves at the domain edges to prevent artificial reflections. 20 points is a standard choice for k-Wave at this frequency. |

---

## Next steps

**Nonlinear propagation.** Setting `medium.BonA = 0.4` enables k-Wave's
Westervelt equation solver, which would capture second-harmonic generation,
sum/difference frequency mixing, and parametric array effects at the beam
intersection. This is a one-line code change. The physical consequence is
that energy would transfer from 40 kHz into 80 kHz, and if the two arrays
were driven at slightly different frequencies, difference-frequency content
in the audible range would appear.

**Absorption.** Adding frequency-dependent atmospheric absorption via
`medium.alpha_coeff` and `medium.alpha_power` would account for the ~1.2 dB/m
loss at 40 kHz, which is small over this domain. More importantly, absorption
at 80 kHz is about 4x stronger, so it would preferentially damp the
second-harmonic content generated by nonlinear propagation.

**3D simulation.** Running the full 8x8 array in 3D would capture focusing
in both transverse dimensions. The focusing gain should roughly double (in dB)
compared to the 2D case. The grid would be approximately 350^3 points instead
of 280 x 350, so the compute cost is substantially higher.

