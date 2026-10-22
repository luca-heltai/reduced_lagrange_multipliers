# MetricFlowX vessel-wall coupling

`two_way.prm.in` is the nonzero-inflow example for the
`metric_flow_x_elastodynamics` application. It configures the flow and solid
Problems, the tensor-product wall lift, the material data, time integration,
initial and boundary functions, and output directories.

The template is generated below the build tree. Run the application with that
generated file, as described in the [tutorial](../../doc/tutorials/metric-flow-x-elastodynamics.md).

The [mathematical formulation](../../doc/tutorials/metric-flow-x-elastodynamics-coupling.md)
is documented in the tutorial documentation.

The [one-vessel stress-jump diagnostic](../../doc/tutorials/metric-flow-x-elastodynamics-mms.md)
derives a continuous, piecewise solid displacement and nonzero pressure-jump
target. Its inlet supplies the changing vessel volume. Run
`mms_cube_diagnostic.prm.in`, `mms_cube_diagnostic_refined.prm.in`, and
`mms_cube_diagnostic_mpi.prm.in` through their generated build-tree `.prm`
files. These inputs expose the current constraint-row mismatch; they are not
passing MMS verification cases. The application does not compute their errors
automatically.
