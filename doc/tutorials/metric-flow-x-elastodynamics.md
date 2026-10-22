# MetricFlowX and elastodynamics

The `metric_flow_x_elastodynamics` application composes a 1D-in-3D
MetricFlowX flow Problem and a 3D elastodynamics Problem in one IDA adapter.
The runnable example uses one parameter file:

```text
tutorials/metric_flow_x_elastodynamics/two_way.prm.in
```

The [one-vessel manufactured solution](metric-flow-x-elastodynamics-mms)
provides a piecewise manufactured target in a cube, with forcing for all five
residual rows, volume supplied by the inlet, and measured field errors.

MetricFlowX must provide the same build configuration as ImmersX.
DebugRelease requires a dual-variant package exporting
`MetricFlowX::metric_flow_x_Debug` and
`MetricFlowX::metric_flow_x_Release`. ImmersX links each library and its
tests to the matching concrete target, keeping the deal.II variants separate.
Older packages exporting only `MetricFlowX::metric_flow_x` remain supported
in compatible single-variant builds; incompatible packages are disabled.
For a Debug-only installation, configure a separate Debug build:

```bash
cmake -S . -B build-metric-debug -DCMAKE_BUILD_TYPE=Debug \
  -DMetricFlowX_DIR="$HOME/dualistic/installations/lib/cmake/MetricFlowX"
cmake --build build-metric-debug -j2
ctest --test-dir build-metric-debug -R MetricFlowX --output-on-failure
```

The configure output must report `MetricFlowX support enabled`. CTest checks
the standalone residual and Jacobians against the native matrices and runs the
nonzero-inflow coupled example in serial and on two MPI ranks, using degree
two to check automatic multiplier selection. Coupled cases write to separate
build-tree output directories.

The inflow starts with $Q(0)=Q'(0)=0$, matching the initially resting solid
and zero initial acceleration. Displacement compatibility also imposes
conditions on the initial velocity and acceleration; a nonzero initial
inflow derivative is incompatible with these zero initial data.

After configuring and building with MetricFlowX and deal.II SUNDIALS enabled,
run the generated build-tree input:

```bash
build/metric_flow_x_elastodynamics_debug \
  build/tutorials/metric_flow_x_elastodynamics/two_way.prm
```

The wall relation is imposed weakly. With (A) the flow area, (A_0) its
reference area, (u) the solid displacement, (n) the radial direction and
(lambda) the pressure-jump multiplier, the lifted kinematic row represents

```text
integral_Gamma (u - (sqrt(A/pi) - sqrt(A0/pi))*n) dot (mu*n) dGamma = 0.
```

The application supplies native external pressure

```text
p_external = -lambda.
```

Native assembly includes volume, face, boundary, and junction contributions.
The solid receives the positive multiplier traction in momentum. All coupling
terms appear together in the application.

The application imports `a0` from the VTK mesh already configured for
MetricFlowX. It represents `a0` as a frozen `FE_DGQ(0)` Observable, so adjacent
cells can keep different reference areas. The same Observable defines both
the radial increment and the lift thickness. The multiplier has an independent
scalar `FiniteElementSpace<1,3>` on the centerline triangulation. Its default
`auto` selection follows the parsed MetricFlowX area degree. The application
checks that the local scalar FE basis matches. The IDA adapter advances both Problems,
and output includes flow, solid, and multiplier fields.
