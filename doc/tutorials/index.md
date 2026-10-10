# Tutorials

These tutorials use the parameter files under `tutorials/` and the
executables listed in the [application reference](../reference/applications).
They cover scalar and vector problems, time integration, reduced geometry,
weak coupling, and a two-way vessel-wall interaction.

The introductory sequence is:

1. [Poisson](poisson) — a scalar finite-element solve.
2. [Static elasticity](elasticity) — a vector-valued finite-element solve.
3. [Elastodynamics](elastodynamics) — displacement and velocity with IDA.
4. [Reduced Poisson](reduced-poisson) — an immersed lower-dimensional space.
5. [Coupled Poisson](coupled-poisson) — two Poisson Problems and a multiplier.
6. [Coupled Poisson–elasticity](coupled-poisson-elasticity) — a lifted load
   from a 1D Problem onto 3D elasticity.
7. [Fiber-reinforced elastodynamics](fiber-reinforced-elastodynamics) — two
   nonmatching Problems and a multiplier constraint.

[Navier–Stokes](navier-stokes) documents the independent incompressible-flow
application. The [MetricFlowX and elastodynamics tutorial](metric-flow-x-elastodynamics)
shows the parameter-driven two-way vessel-wall coupling.

```{toctree}
:maxdepth: 1

poisson
elasticity
elastodynamics
reduced-poisson
coupled-poisson
coupled-poisson-elasticity
fiber-reinforced-elastodynamics
navier-stokes
metric-flow-x-elastodynamics
```
