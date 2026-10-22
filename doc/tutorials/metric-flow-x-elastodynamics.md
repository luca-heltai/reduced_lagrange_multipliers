# MetricFlowX and elastodynamics

The `metric_flow_x_elastodynamics` application composes a 1D-in-3D
MetricFlowX flow Problem and a 3D elastodynamics Problem in one IDA adapter.
The runnable example uses one parameter file:

```text
tutorials/metric_flow_x_elastodynamics/two_way.prm.in
```

The [one-vessel stress-jump diagnostic](metric-flow-x-elastodynamics-mms)
provides a piecewise manufactured target in a cube. Its separate parameter
files expose differences between that target and the current constraint
assembly; they are not passing MMS verification cases.

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

The application assembles the flow feedback as a cell-volume WeakTerm using
the MetricFlowX velocity extractor:

```text
integral_centerline -lambda * v_U ds.
```

This term uses values on cells. It does not add DG face fluxes. The multiplier
convention remains `lambda = p_external - p_internal`; the solid receives the
positive multiplier reaction, and the flow residual receives the negative
pairing above.

The application imports `a0` from the VTK mesh already configured for
MetricFlowX. It represents `a0` as a frozen `FE_DGQ(0)` Observable, so adjacent
cells can keep different reference areas. The same Observable defines both
the radial increment and the lift thickness. The multiplier has an independent
scalar `FiniteElementSpace<1,3>` on the centerline triangulation. Its default
`auto` selection follows the parsed MetricFlowX area degree. The application
checks that the local scalar FE basis matches. The IDA adapter advances both Problems,
and output includes flow, solid, and multiplier fields.
