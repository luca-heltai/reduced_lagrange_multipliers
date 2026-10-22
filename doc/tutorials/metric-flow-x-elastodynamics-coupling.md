# MetricFlowX and elastodynamics coupling

This page describes the equations assembled by the
`metric_flow_x_elastodynamics` application. The model couples a 1D-in-3D
MetricFlowX blood-flow Problem to a 3D elastodynamics Problem through the
vessel wall. Both Problems and the wall interaction are added to one IDA
execution adapter.

## State variables

Write the solid displacement and velocity as $u(x,t)$ and $v(x,t)$.
The MetricFlowX state contains at least the cross-sectional area $A(s,t)$
and axial flow velocity $U(s,t)$, where $s$ is the vessel centerline
coordinate. The wall interaction adds a scalar multiplier field
$\lambda(s,t)$ on that same one-dimensional centerline.

The elastodynamics Problem uses the first-order semidiscrete form

$$
  M(\dot u-v)=0,\qquad
  M\dot v + K u + C v - f=0,
$$

where $M$, $K$, and $C$ are its mass, stiffness, and damping
operators, and $f$ is the body-force vector. MetricFlowX supplies its native
flow residual $F_{\mathrm{flow}}(A,U,\ldots)$.

## Area-to-wall kinematics

The resting area $A_0$ comes from the MetricFlowX vessel metadata. Define

$$
  R_0=\sqrt{A_0/\pi},\qquad
  \delta R(A)=\sqrt{A/\pi}-R_0,\qquad
  P(A)=\delta R(A)n,
$$

where $n$ is the unit radial direction on the reconstructed wall. Thus
$P(A_0)=0$. The tensor-product lift uses cross-section modes 3 and 7 to
represent this radial direction. It transfers the centerline area and
multiplier fields to the wall using the configured cross-section quadrature
and source thickness.

The kinematic condition is imposed weakly. For every scalar centerline test
function $\mu_h$, its lift $\mu_h^\ell n$ tests the radial displacement
mismatch:

$$
  \int_{\Gamma} u_h\cdot(\mu_h^\ell n)\,d\Gamma
  -
  \int_{\Gamma} P(A_h)\cdot(\mu_h^\ell n)\,d\Gamma=0.
$$

Equivalently, the second integrand is $\delta R(A_h)\mu_h^\ell$. These
integrals describe the continuous pairing; the discrete assembly uses the
wall lift's quadrature, mode reconstruction, and geometric weights. The
constraint is enforced through these lifted weak pairings.

In operator notation, the assembled constraint is

$$
  B^T u_h-P_h(A_h)=0\quad\text{in }\Lambda_h^*,
$$

where $\Lambda_h$ is the scalar finite-element space for $\lambda_h$,
and $\Lambda_h^*$ is its dual. The assembled residual is a vector with one
scalar entry per multiplier degree of freedom. Both terms in that residual
are dual-space vectors, even though the underlying displacement and wall
map are vector-valued fields on the wall.

## Multiplier and pressure signs

The multiplier is a scalar pressure field, not a single spatially constant
number. Its interpretation is the pressure jump

$$
  \lambda=p_{\mathrm{ext}}-p_{\mathrm{int}}^{\mathrm{fitt}},
$$

between the external pressure and the internal reference pressure used in
the wall/flow convention. The multiplier couples the solid and flow residuals
with opposite signs:

$$
  F_{\mathrm{solid}}+B\lambda=0,
  \qquad
  F_{\mathrm{flow}}(A,U,\ldots)+F_{\Delta p}(\lambda)=0.
$$

In the current assembly, positive $\lambda$ enters the solid row through
$+B\lambda$, in its momentum equation. The flow receives $p_{\mathrm{external}}=-\lambda$
through native pressure assembly, including its volume, numerical flux,
boundary, and junction terms. Native assembly supplies the density scaling.

## Coupled residual

Collecting the solid state, flow state, and multiplier into $y$, the coupled
semidiscrete problem has the schematic residual form

$$
  F(t,y,\dot y)=
  \begin{bmatrix}
    M(\dot u-v)\\
    M\dot v+Ku+Cv-f+B\lambda\\
    F_A^{\mathrm{native}}\\
    F_U^{\mathrm{native}}+F_U^{\mathrm{pressure}}(-\lambda)\\
    B^T u-P_h(A)
  \end{bmatrix}=0.
$$

The area affects the solid through the nonlinear radius map in the kinematic
constraint. The multiplier balances solid momentum and supplies native external
pressure. Algebraic pressure trace equations also receive this pressure.
Physical continuity contains no multiplier reaction. Differentiating
compatibility gives $B^Tv-DP_h(A)\dot A=0$, which connects wall velocity
to the area rate without replacing the displacement constraint.

The application registers the multiplier and shows compatibility, traction,
and native pressure terms together. The generic symmetric constraint would
add transpose reactions to displacement and area residuals, which are not the momentum
rows of this first-order system.

The executable requires MetricFlowX, deal.II with SUNDIALS, and MPI. The
application and its build requirements are listed in the
[application reference](../reference/applications).
