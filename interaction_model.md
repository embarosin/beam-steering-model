# Beam Interaction Model

## Primary interaction: linear superposition

Both arrays are run simultaneously in a single k-Wave simulation. The solver
naturally computes the full wave interaction -- the pressure fields from both
arrays propagate, interfere, and superpose exactly as they would physically.
This is more accurate than post-hoc addition of separate fields because the
time-domain solver captures:

- Constructive/destructive interference at every grid point
- The correct temporal evolution of the combined field
- Boundary reflections (absorbed by PML) from the combined field

At the focal point, where both beams arrive in-phase by construction, the
combined field is +18.3 dB above the unfocused baseline. The combined focused
field is ~+6 dB above a single focused array (103.9 vs 98.2 dB), consistent
with coherent addition of two equal-amplitude sources: 20 * log10(2) = 6.02 dB.


## What linear superposition does not capture

The main omission is nonlinear acoustic effects, which matter when pressure
amplitudes become non-negligible compared to the ambient pressure. Two
phenomena are relevant here:

### Harmonic generation and parametric mixing

At the beam intersection, where the combined amplitude is highest, the
nonlinear term in the wave equation (proportional to p^2) generates:

- Second harmonics at 80 kHz from each beam individually
- A sum-frequency component at 80 kHz from the cross-interaction
- A difference-frequency component at 0 Hz (a DC radiation pressure)

The difference-frequency mechanism is the basis of the parametric array effect
(Berktay), where two ultrasonic beams mix to produce a directional
audible-frequency beam. At the pressure levels in this simulation (~4 Pa peak),
these effects would be present but small.

### Acoustic streaming

Time-averaged momentum transfer from the acoustic field to the medium drives
a steady bulk flow (Eckart streaming). At the beam intersection, this produces
a localized flow of air in the beam propagation direction. Streaming velocity
scales with the square of the pressure amplitude and inversely with viscosity.
For focused 40 kHz fields in air, rough estimates give flow speeds on the order
of cm/s -- enough to:

1. Advect the wavefronts from both arrays, slightly Doppler-shifting the
   apparent frequency at the focal region.
2. Create a convective acoustic non-reciprocity: waves propagating into the
   stream vs. against it accumulate different phases.
3. Modify the local effective sound speed, subtly detuning the constructive
   interference condition at the focal point.


## How nonlinear effects could be included

k-Wave supports nonlinear propagation through the medium's B/A parameter.
Setting `medium.BonA = 0.4` (the value for air) would enable the Westervelt
equation solver, capturing harmonic generation and parametric mixing without
any additional code beyond setting the medium property. The physical
consequence is that energy would transfer from 40 kHz into 80 kHz, and if the
two arrays were driven at slightly different frequencies, difference-frequency
content in the audible range would appear.
