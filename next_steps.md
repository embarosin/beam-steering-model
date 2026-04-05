# What I'd Do Next

With another two hours I would push in three directions. First, I'd enable
nonlinear propagation by setting `medium.BonA = 0.4` (the B/A parameter for
air) -- k-Wave natively supports the Westervelt equation in its time-domain
solver, so this would capture second-harmonic generation, sum/difference
frequency mixing, and parametric array effects at the beam intersection without
any additional code beyond setting the medium property. Second, I'd add
thermoviscous absorption via `medium.alpha_coeff` and `medium.alpha_power` --
at 40 kHz the absorption is ~1 dB/m in air, modest over our 30 cm path but
significant at the 80 kHz second harmonic (~4 dB/m), and k-Wave's
frequency-dependent absorption model handles this naturally. Third, I'd run
the full 8x8 array in a 3D simulation to capture focusing in both transverse
dimensions -- the focusing gain should roughly double (in dB) compared to the
2D case, though the grid would be approximately 350^3 points instead of
280 x 350, so the compute cost is substantially higher.
