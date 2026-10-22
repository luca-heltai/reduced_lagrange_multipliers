# One vessel in a cube: a stress-jump diagnostic

This example prescribes a nonzero, time-dependent radial displacement, vessel
area, axial velocity, and pressure-jump multiplier. A continuous, piecewise
solid displacement supplies the stress jump needed to balance the multiplier.
The inlet supplies the increase in vessel volume; physical continuity has no
distributed mass source.

The inputs run the existing `metric_flow_x_elastodynamics` application without
changing its code. They are a **diagnostic of the current assembly**, not a
passing manufactured-solution verification. The application puts the solid
constraint reaction in the displacement/kinematic row and also adds a
transpose reaction to the flow-area row. The analytical target below exposes
both differences. Successful time integration does not establish that the
computed fields match this target.

![Analytical radial displacement, axial velocity, and volume supplied by the inlet](../_static/metric-flow-x-mms-target.svg)

The plots show the analytical target. No numerical solution is plotted.

## Configure and run

The parameter templates are in `tutorials/metric_flow_x_elastodynamics/`:

| Template | Solid cells | Purpose |
| --- | ---: | --- |
| `mms_cube_diagnostic.prm.in` | $4\times16\times16$ | Single-rank diagnostic |
| `mms_cube_diagnostic_refined.prm.in` | $4\times32\times32$ | Refine the transverse solid mesh |
| `mms_cube_diagnostic_mpi.prm.in` | $4\times16\times16$ | Separate output directories for MPI |

All cases use one centerline cell, degree-one elements, an automatically
matched multiplier space, and the same analytical target. The solid varies
only in the transverse coordinates, so the refined input keeps the axial
partition fixed. The cross-section refinement level is five in all inputs.
The target area is constant in space and the target velocity is linear, so
degree one represents both exactly on that centerline cell. Keep flow
refinement at zero: the application imports `a0` from the original VTK mesh
after flow setup and requires matching active cells.

Configure with a compatible MetricFlowX package and deal.II with SUNDIALS and
VTK enabled. For example, from the repository root:

```bash
cmake -S . -B build-mms -DCMAKE_BUILD_TYPE=Release \
  -DMetricFlowX_DIR=/path/to/metric-flow-x/lib/cmake/MetricFlowX
cmake --build build-mms --target metric_flow_x_elastodynamics -j2
```

Require `MetricFlowX support enabled` in the configuration output. Pass
`DEAL_II_DIR` as well if deal.II is not already discoverable. A Debug build
uses the target and executable `metric_flow_x_elastodynamics_debug`.

CMake generates the runtime inputs, including the absolute include path in
the refined and MPI cases. Run the generated `.prm` files from the build
directory so that the application's `last_used_parameters.prm` also stays
there:

```bash
cd build-mms
./metric_flow_x_elastodynamics \
  tutorials/metric_flow_x_elastodynamics/mms_cube_diagnostic.prm
./metric_flow_x_elastodynamics \
  tutorials/metric_flow_x_elastodynamics/mms_cube_diagnostic_refined.prm
mpirun -np 2 ./metric_flow_x_elastodynamics \
  tutorials/metric_flow_x_elastodynamics/mms_cube_diagnostic_mpi.prm
```

Reconfigure after changing a template. These inputs select one spatial solve
each; `Number of refinement cycles` does not make this coupled executable
perform a convergence study. IDA controls its own time steps. `Fixed step`
parameters do not set the IDA step size.

## Geometry and coefficients

The solid fills the cube $\Omega=[-\ell,\ell]^3$, with
$\ell=0.120685$. The straight centerline is
$\gamma=\{(x,0,0):-\ell\le x\le\ell\}$, of length $L=2\ell=0.24137$.
The existing `data/metric_flow_x/single_vessel_centered.vtk.in` supplies

$$
A_0=a_d=3.0605\times10^{-4},\qquad
R_0=\sqrt{A_0/\pi}\simeq0.00987009324508.
$$

No new mesh file is needed. The fictitious solid inside the vessel is part of
the cube. The immersed reference wall is
$\Gamma_0=\{r=R_0\}$, where $r=\sqrt{y^2+z^2}$. Its radius is fixed in the
lift; the current area enters the displacement increment.

The solid density and Lamé coefficients are
$\rho_s=\lambda_s=\mu_s=1$, with zero damping. These are verification
coefficients. Flow density is $1060$, viscosity is zero, and the existing VTK
tube-law properties are retained. The numerical time interval is $0\le t\le
T=0.1$.

## Piecewise solid solution

Let $g(t)=t^3$ and define the spatial vector

$$
w(x,y,z)=
\begin{cases}
(0,y,z), & r\le R_0,\\[2pt]
\displaystyle\frac{R_0^2}{r^2}(0,y,z), & r>R_0.
\end{cases}
$$

The target displacement, velocity, and body force are

$$
u^\star=t^3w,\qquad
v^\star=3t^2w,\qquad
f^\star=6\rho_s t\,w.
$$

The displacement is continuous at $r=R_0$ and regular at the centerline.
Its radial derivative jumps at the wall. Away from the wall, both $r$ and
$R_0^2/r$ solve the radial homogeneous elasticity equation, so
$\operatorname{div}\sigma(u^\star)=0$ in each subdomain. Thus the body force
above supplies the acceleration without a volume approximation to a surface
force.

For

$$
\sigma(u)=2\mu_s\varepsilon(u)+\lambda_s\operatorname{div}(u)I,
$$

the radial stresses at the interface are

$$
\sigma_{rr}^{\mathrm{in}}=2(\lambda_s+\mu_s)g,\qquad
\sigma_{rr}^{\mathrm{out}}=-2\mu_s g.
$$

With $n$ directed from the centerline toward the surrounding solid, the jump
is $[\sigma n]=(\sigma^{\mathrm{in}}-\sigma^{\mathrm{out}})n$.
The momentum balance

$$
(\rho_s\dot v,\phi)_\Omega+
(\sigma(u),\varepsilon(\phi))_\Omega-
(f,\phi)_\Omega+
\int_{\Gamma_0}\lambda\,n\cdot\phi\,d\Gamma-
\int_{\partial\Omega}\tau\cdot\phi\,dS=0
$$

therefore requires

$$
\lambda^\star=-2(\lambda_s+2\mu_s)g=-6t^3.
$$

This is the scalar pressure-jump multiplier. Its negative sign balances the
outward displacement under the positive solid-reaction convention.

All cube faces have boundary id zero. Prescribe the exact outer traction
$\tau^\star=\sigma(u^\star)n_\Omega$ on every face, including the vessel end
planes. Both initial fields are zero.
The initial acceleration and multiplier are also zero, matching the
application's initialization. The parameter expression uses
$R_0^2/\max(y^2+z^2,R_0^2)$ to encode both branches without division by zero.

For $r<R_0$, the stress matrix is
$t^3\operatorname{diag}(2,4,4)$. For $r>R_0$, its nonzero components are

$$
\sigma_{yy}=\frac{2t^3R_0^2(z^2-y^2)}{r^4},\qquad
\sigma_{zz}=-\sigma_{yy},\qquad
\sigma_{yz}=\sigma_{zy}=-\frac{4t^3R_0^2yz}{r^4}.
$$

The boundary expression multiplies this matrix by the appropriate cube-face
normal. It uses a coordinate tolerance of $10^{-10}$ to identify faces and
regularizes denominators in branches where the numerator is zero. No
Dirichlet ids are selected. The transient mass operator and zero initial
displacement/velocity fix the rigid-body motion of this traction problem.

This boundary choice also keeps the lift's affine constraints fixed. A trial
with time-dependent exact Dirichlet data stops with `Lifted affine constraints
changed after static assembly; rebuild the weak term.` The supplied inputs
use exact traction to avoid that separate limitation.

## Area and volume supplied by the inlet

On the reference wall, $u^\star\cdot n=R_0g$. The area-to-radius relation
therefore gives the spatially uniform target

$$
A^\star(t)=A_0(1+t^3)^2,\qquad
\dot A^\star(t)=6A_0t^2(1+t^3).
$$

Let $s=x+\ell\in[0,L]$. Use

$$
Q^\star(s,t)=(L-s)\dot A^\star(t),\qquad
U^\star(x,t)=(\ell-x)\frac{6t^2}{1+t^3}.
$$

These fields satisfy source-free continuity pointwise:

$$
\partial_t A^\star+\partial_s(A^\star U^\star)=0.
$$

At the endpoints,

$$
Q_{\mathrm{in}}^\star=L\dot A^\star,\qquad
Q_{\mathrm{out}}^\star=0,\qquad
\frac{d}{dt}(LA^\star)=Q_{\mathrm{in}}^\star-Q_{\mathrm{out}}^\star.
$$

The inlet function is this nonzero, time-dependent volumetric flow rate.
The outlet uses `Reflection` with `Rt = 1`. The native characteristic
equations then impose zero outlet velocity. Zero outlet flux is compatible
with expansion because the inlet supplies all of the added volume. Setting
both endpoint fluxes to zero would require a distributed mass source.

For this target, the tube-law pressure is uniform along the vessel. With zero
friction, the native momentum equation and the application's explicit
cell-volume feedback give

$$
\partial_t U+U\partial_s U+\rho_f^{-1}\partial_s p-\lambda=S_U.
$$

Consequently the two RHS expressions are

$$
S_A=0,\qquad
S_U=(\ell-x)\frac{12t-42t^4}{(1+t^3)^2}+6t^3.
$$

The last term balances the implemented $-\lambda$ pairing. No pressure
gradient or density factor is inserted into that pairing. This is an
algebraic diagnostic of the existing residual scaling. It does not replace
that term with a different pressure model.

## What the current assembly changes

Write the discrete wall constraint as
$C(u,A)=B^Tu-P_h(A)=0$. The generic `Constraint` adds a transpose reaction to
each participating field. In the current application these are displacement
and the area component of the mixed flow state. The relevant rows are

$$
\begin{aligned}
F_u &= M(\dot u-v)+B\lambda,\\
F_v &= M\dot v+Ku-f,\\
F_A &= F_A^{\mathrm{native}}-DP_h(A)^T\lambda,\\
F_U &= F_U^{\mathrm{native}}-M_\gamma\lambda,\\
F_\lambda &= B^Tu-P_h(A).
\end{aligned}
$$

Here $M_\gamma$ denotes the centerline value pairing. The explicit flow
feedback adds the fourth-row term. It does not remove the area reaction
created by `Constraint`. The assembled solid load $f$ includes the body force
and outer Neumann traction.

The solid target has $\dot u^\star=v^\star$, so its first row contains the
nonzero residual $B\lambda^\star$. Its stress jump also requires a reaction in
the momentum row, where the current assembly has none. A parameter-defined
body force enters only that momentum row and cannot move the constraint
reaction between rows.

The area reaction is also visible in the continuous circular-wall limit:

$$
\delta R'(A)=\frac{1}{2\sqrt{\pi A}},\qquad
2\pi R_0\delta R'(A^\star)=\frac{1}{1+g},
$$

so the assembled area equation corresponds to

$$
\partial_t A+\partial_s(AU)-\frac{\lambda}{1+g}=S_A.
$$

At the analytical target, its extra term is $6t^3/(1+t^3)$. The inputs keep
$S_A=0$ so that this discrepancy remains visible. Adding an equal RHS source
would mask it. For computed fields, use the actual derivative of the radius
map and the actual multiplier; substituting $1+g$ assumes the target area.
Finite cross-section geometry and quadrature introduce additional differences
from the circular-wall formula.

The row placement follows `ElastodynamicsFields` in
`include/immersx/physics/elastodynamics_semidiscrete.h`, the field selection in
`apps/app_metric_flow_x_elastodynamics.cc`, and the participant reactions in
`WeakTerm::add_constraint_terms()` in `include/immersx/core/weak_term.h`.
No executable, library, or build-system source is changed by this example.

## Inspect the diagnostic

Outputs are below the build-tree `test_output/` directory, in separate
`metric-flow-x-mms-cube`, `metric-flow-x-mms-cube-refined`, and
`metric-flow-x-mms-cube-mpi` directories. Each contains solid output and a
`flow/interaction/pressure_jump.pvd` multiplier series. Flow output contains
`area`, `velocity`, `A_hat`, `U_hat`, and `pressure`.

At the final target time $T=0.1$:

| Quantity | Analytical target |
| --- | ---: |
| Area $A^\star$ | $3.0666240605\times10^{-4}$ |
| Radial wall increment $R_0T^3$ | $9.87009324508\times10^{-6}$ |
| Pressure jump $\lambda^\star$ | $-0.006$ |
| Inlet velocity $U^\star(-\ell,T)$ | $0.0144677322677$ |
| Inlet flow $Q_{\mathrm{in}}^\star(T)$ | $4.43670958731\times10^{-6}$ |
| Outlet flow $Q_{\mathrm{out}}^\star(T)$ | $0$ |

In ParaView, compare the solid radial displacement along a transverse line
with $t^3r$ inside and $t^3R_0^2/r$ outside. Compare the centerline area with
$A_0(1+t^3)^2$, axial velocity with $(\ell-x)6t^2/(1+t^3)$, and
`pressure_jump` with $-6t^3$. Use the accepted physical time, not a file index.

The current application leaves the native flow clock at its initial value
when it calls the flow writer. Consequently every timestamp in
`flow/stress_jump.pvd` is zero. Match flow file indices to the accepted times
in `flow/interaction/pressure_jump.pvd` or the solid series. Do not use the
flow PVD timestamps to evaluate the analytical functions. The application
writes the initial state twice; duplicate step-zero entries are not two time
steps.

Check the integrated volume balance independently:

$$
V_h(t)=\int_\gamma A_h\,ds,\qquad
\frac{dV_h}{dt}\mathrel{?}=Q_{\mathrm{in},h}-Q_{\mathrm{out},h}.
$$

For the target, the increase is
$V^\star(t)-V^\star(0)=LA_0[(1+t^3)^2-1]$. For computed fields, report any
area-reaction contribution separately. A centered difference between outputs
only approximates the volume derivative; distinguish that error from a
conservation defect.

The application does not call either Problem's error routine or produce
coupled error tables. The `Exact solution` entries store the target formulas;
they do not activate error computation. Compare exported fields externally.
Do not claim convergence to this target from successful solver exits or from
agreement between serial and MPI runs. The piecewise stress field, unfitted
solid mesh, cross-section geometry, and time error must also be considered
when interpreting differences between the two meshes.
