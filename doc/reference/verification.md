# Verification inputs

The repository keeps manufactured-solution and convergence inputs alongside
the tutorial assets so they remain runnable without becoming part of the
introductory learning path.

- `tutorials/elasticity/` contains immersed-elasticity static and dynamic
  verification cases.
- `tutorials/elastodynamics/` contains standalone time-dependent convergence
  cases.
- `gtests/parameters/` contains small configured inputs used by focused tests.

The MetricFlowX–elastodynamics example is configured by
`tutorials/metric_flow_x_elastodynamics/two_way.prm.in` and documented in one
[tutorial](../tutorials/metric-flow-x-elastodynamics). Its focused coupling
regressions cover the Area-to-radius Observable, the lifted kinematic row, the
cell-volume multiplier pairing against the MetricFlowX velocity extractor, and
the coupled Jacobian. The parameter-driven example exercises the transient
adapter path and writes the flow, solid, and multiplier fields.

Use the [application reference](applications) to choose the executable and the
[testing guide](../developer/testing) to run a regression or GoogleTest.
