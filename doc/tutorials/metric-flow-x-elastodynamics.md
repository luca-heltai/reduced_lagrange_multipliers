# MetricFlowX and elastodynamics

The `metric_flow_x_elastodynamics` application composes a 1D-in-3D
MetricFlowX flow Problem and a 3D elastodynamics Problem in one IDA adapter.
The runnable example uses one parameter file:

```text
tutorials/metric_flow_x_elastodynamics/two_way.prm.in
```

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
the radial increment and the lift thickness. The multiplier is a compacted
Field exported on the parent centerline FE space. The IDA adapter advances
both Problems, and output includes flow, solid, and multiplier fields.
