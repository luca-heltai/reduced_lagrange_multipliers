# Application coupling roadmap

This roadmap defines the application examples that will exercise ImmersX's
coupling API. It separates two questions:

- what mathematical relation connects the participating fields; and
- how those fields are represented geometrically.

The first question is described by the coupling patterns P1--P3. The second
is described by the geometry levels G0--G3. They are independent axes. A
change from one geometry level to another should normally change field or
observable construction and backend evaluation, not the mathematical weak
terms that express the relation.

This page is the roadmap for the application stack. The status table records
the intended acceptance status of each matrix entry; existing applications
listed below are related starting points, not evidence that the corresponding
roadmap entry is complete.

## Coupling patterns

### P1: prescribed one-way forcing

Problem A is active. Data from another discretization is prescribed input:

```text
A(u) = F(b_given)
```

The active field `u` must remain distinct from the prescribed field or frozen
observable `b_given`. There is no second active Problem and no multiplier. The
application-facing target is an ordinary weak term, for example:

```cpp
auto b = imported_field(...); // target syntax; use the current import API
weak_term(F(b), test(u));
```

The current public import path uses an
`ImportedFiniteElementFields` object and `frozen(field, coefficients)`. A
roadmap application must use those existing facilities or extend them only
when the example exposes a concrete missing operation.

### P2: prescribed one-way constraint

Problem A is active, and a multiplier enforces a relation to prescribed data:

```text
A(u) + C'(u)^T lambda = f
C(u) - F(b_given) = 0
```

The active side `C(u)` and prescribed side `F(b_given)` have different
semantics. The prescribed side is not represented by a fake active Problem.
The multiplier is an auxiliary algebraic field and its output represents the
reaction required to enforce the target.

### P3: two live coupled Problems

Both Problems own active state fields:

```text
A(u) + C'(u)^T lambda = f
E(w) - D'(w)^T lambda = g
C(u) - D(w) = 0
```

The intended user-facing shape is a `Constraint` made from the two
participant expressions:

```cpp
auto relation = make_constraint(
  weak_term(C(u), test(lambda)) -
  weak_term(D(w), test(lambda)));
```

The exact expression depends on the participating fields. The important
properties are that `u` and `w` are active fields, `lambda` is an independent
multiplier field, and the constraint produces reactions on both Problems.

## Geometry levels

### G0: same dimension, same grid

This is the baseline for API semantics. The fields may use the same
triangulation and DoFHandler, or independent DoFHandlers on the same
triangulation. No nonmatching search or lifting is needed.

### G1: same dimension, nonmatching grids

The fields have the same physical dimension but independent meshes and
DoFHandlers. The evaluation and assembly backend must handle the nonmatching
geometry. The variational form should remain the same as at G0.

The first physically meaningful G1 case is tied elasticity. It uses a tied
relation, not contact or friction inequalities.

### G2: mixed dimension, direct evaluation

The fields live on geometrically embedded objects with different dimensions,
such as a 3D bulk and a 2D surface or 1D fiber. The lower-dimensional quantity
is evaluated directly on the pairing support. It is still a direct
evaluation/search problem; it does not require a reconstructed bulk quantity.

Reduced-dimensional ownership and point search must use the relevant
distributed deal.II facilities. A lower-dimensional field is not assumed to
be serial.

### G3: mixed dimension with lift or reconstruction

The reduced-dimensional observable is transformed into the representative
physical quantity before it is used:

```text
reduced Field -> Observable -> nonlinear transform -> lift/reconstruction
               -> physical quantity -> WeakTerm or Constraint
```

The lift is an observable or backend operation inside the relevant P1, P2, or
P3 application. It is not a fourth coupling pattern and must not become a
second application-facing coupling hierarchy.

## The application matrix

The matrix is architectural coverage, not a request for twelve duplicate
programs. A single application may cover more than one cell when the shared
physics keeps the comparison clear.

| Pattern | G0: same grid | G1: nonmatching grids | G2: mixed dimension/direct | G3: mixed dimension/lift |
| --- | --- | --- | --- | --- |
| P1 prescribed forcing | validated | validated | validated | validated |
| P2 prescribed constraint | validated | validated | validated | validated |
| P3 two live Problems | validated | validated | validated | flagship |

The status values used here are:

- `planned`: identified in the stack, but not accepted as a roadmap example;
- `implemented`: the application exists and exercises the intended concept;
- `validated`: focused correctness and the relevant application checks pass;
- `flagship`: a physically meaningful acceptance application for the geometry
  and coupling combination.

The current evidence for the promoted entries is deliberately small. The
`ApplicationRoadmap` tests execute the three G0 patterns with real Poisson
Problems, including frozen forcing, a prescribed right-hand side, a two-Problem
constraint, and solution output. The same suite validates P1/G1 with frozen
forcing on an independent same-dimensional mesh and P2/G1 with a prescribed
constraint on an independent same-dimensional mesh. It also validates P1/G2
with a full-dimensional frozen field forcing an embedded line Problem. The
same suite validates P1/G3 with a one-way frozen line load transformed by a
tensor-product lift and P2/G3 with prescribed line motion enforced through a
lifted multiplier test. It also validates P3/G1 with two independent
distributed elasticity Problems on offset grids, a vector multiplier,
participant metadata, reactions, and output. The `PrescribedPoisson` tests
validate P2/G2 with an embedded prescribed target, a multiplier, and output.
The `CoupledPoisson` tests implement a mixed-dimensional P3/G2 precursor with
two active Problems and a multiplier. The roadmap suite now also validates
the P3/G2 fiber-elastodynamics shape with independent 2D matrix and embedded
1D fiber Problems, a generic IDA adapter, a nonzero line-multiplier reaction,
and output; its serial and two-rank MPI tests are separate from the quick
serial gate. The
existing `FiberReinforcedElastodynamics` driver uses the same
mixed-dimensional matrix-plus-line-fiber composition. MetricFlowX plus
elastodynamics remains the
P3/G3 flagship target; its focused validation and application tests remain
separate from the quick roadmap gate. The P3/G1 gate also passes with two MPI
ranks, exercising distributed nonmatching execution.

P3/G1 nonmatching tied elasticity, P3/G2 FiberReinforcedElastodynamics, and
P3/G3 MetricFlowX plus elastodynamics are the three flagship targets. The
remaining cells are deliberately smaller examples whose purpose is to isolate
one semantic or geometric step. The P3/G3 flagship is covered by focused
MetricFlowX integration tests and a single parameter-driven two-way
application example. The coupling uses a generic kinematic constraint and
cell-volume pressure feedback.

## Foundational examples

These examples clarify the boundary between ordinary equation coupling and a
genuine constrained relation. They are outside the 3 x 4 matrix.

### F1: mixed fourth-order problem

Use a mixed formulation of a biharmonic problem. With `w = Delta u`, one
representative sign convention is:

```text
Delta u - w = 0
-Delta w = f
```

The signs and boundary conditions must be chosen consistently by the concrete
example. `u` and `w` are active fields, and the two equations are connected by
ordinary cross-equation `WeakTerm`s. No Lagrange multiplier or `Constraint` is
needed merely because the Problem has multiple fields.

This is the canonical example for a small multi-field Problem and block
execution. It should make the following distinction explicit:

```text
WeakTerm   ordinary equation coupling
Constraint constrained relation with an auxiliary reaction field
```

The current minimal test uses ordinary terms directly:

```cpp
const auto u = V.field(layout, "u");
const auto w = V.field(layout, "w");

weak_term(gradient(u), gradient(test(w))).add(builder);
weak_term(value(w), test(w)).add(builder);
weak_term(gradient(w), gradient(test(u))).add(builder);
```

### F2: incompressible elasticity

Use the small saddle-point form

```text
A(u) + B^T p = f
B u = 0
```

Here `u` is a vector field and `p` is a pressure-like multiplier field. This
is a physical example of a saddle point and constraint-style structure. It is
intentionally small; its purpose is to explain the field and block roles, not
to become a large mechanics application.

The corresponding current API remains an ordinary weak-term composition, with
explicit saddle-point metadata for the pressure field:

```cpp
const auto displacement =
  V.field(layout, "displacement", FEValuesExtractors::Vector(0));
const auto pressure =
  V.field(layout, "pressure", FEValuesExtractors::Scalar(2));

weak_term(symmetric_gradient(displacement),
          symmetric_gradient(test(displacement)))
  .add(builder);
weak_term(value(pressure), divergence(test(displacement))).add(builder);
weak_term(divergence(displacement), test(pressure)).add(builder);
builder.saddle_point(pressure.field_id(), {displacement.field_id()});
```

## Physical examples by geometry level

The simple examples isolate API semantics. The more physical examples are
reserved for the geometry levels where they provide a useful acceptance test.

### G0 matching examples

- **P1: trivial Poisson imported forcing.** Solve `-Delta u = F(b_given)` on a
  matching grid. Demonstrate imported or frozen data, an `Observable`, and a
  `WeakTerm` without a multiplier.
- **P2: prescribed elasticity constraint.** Solve an elasticity Problem with a
  prescribed displacement or velocity target on the same grid. Output the
  multiplier reaction.
- **P3: tied Problems on one grid.** Use two independent Poisson-like or
  elasticity-like Problems with active fields and a multiplier enforcing
  `C(u_A) - D(u_B) = 0`. This must not be replaced by F1.

### G1 nonmatching same-dimensional examples

- **P1: Poisson forcing from an independent mesh.** Keep the physics trivial
  and change only the source mesh and nonmatching evaluation path.
- **P2: prescribed motion from an independent mesh.** Use a prescribed
  displacement or velocity target on an independent same-dimensional mesh.
- **P3: nonmatching tied elasticity.** Couple two independent elastic bodies
  with a tied or mortar-style relation. This is the first physically
  meaningful nonmatching two-Problem case and must test participant reactions,
  transpose consistency, and distributed execution.

Contact inequalities and friction are outside this roadmap.

### G2 direct mixed-dimensional examples

- **P1: embedded source.** Force a bulk diffusion or Poisson-like Problem
  with data on an embedded line, surface, or network.
- **P2: prescribed immersed motion.** Enforce prescribed displacement or
  velocity on an embedded lower-dimensional object with a multiplier.
- **P3: FiberReinforcedElastodynamics.** Couple active matrix and fiber
  elastodynamics through a kinematic multiplier relation. The application is
  the main P3/G2 acceptance test: it must use independent semantic fields,
  mixed-dimensional direct evaluation, time integration, participant
  reactions, and multiplier output.

### G3 lifted mixed-dimensional examples

- **P1: one-way lifted vessel load.** Transform prescribed reduced-dimensional
  pressure or area data into a lifted representative load on a structural
  Problem.
- **P2: prescribed vessel-wall kinematics.** Transform prescribed `A(s,t)` to
  a radius, lift the radius to a wall displacement, and impose that target by
  a multiplier.
- **P3: MetricFlowX plus elastodynamics.** Couple active MetricFlowX and
  structural elastodynamics. Use a generic kinematic `Constraint`; keep
  MetricFlowX-specific pressure or feedback behavior in an `Interaction` only
  where that behavior is genuinely physics-specific. MetricFlowX remains an
  external Problem and is not modified by this roadmap.

The G2 and G3 examples must output each primary physical field and any
multiplier fields. Application code must not need global execution block
numbers to write those fields.

## Implementation order

The stack is intentionally progressive:

1. **PR 1 — this roadmap.** Establish the patterns, geometry levels, matrix,
   foundational examples, and implementation order.
2. **PR 2 — matching foundations.** Implement F1, F2, and the smallest G0
   examples needed to show the public API with real compilable code.
3. **PR 3 — complete G0.** Implement P1/G0, P2/G0, and P3/G0 using simple
   physics and a real two-Problem `Constraint` for P3.
4. **PR 4 — nonmatching G1.** Reuse the G0 semantics while adding
   same-dimensional nonmatching evaluation, ending with tied elasticity.
5. **PR 5 — prescribed G2.** Add the embedded source and prescribed immersed
   motion precursors.
6. **PR 6 — FiberReinforcedElastodynamics.** Bring the flagship G2/P3
   application into the generic Field/Observable/WeakTerm/Constraint/
   ExecutionAdapter model without changing protected production Problems.
7. **PR 7 — prescribed G3.** Add the one-way lifted load and prescribed
   vessel-wall kinematics examples.
8. **PR 8 — MetricFlowX coupling.** Complete the active G3/P3 application,
   including the generic kinematic constraint and any necessary
   physics-specific interaction.

The implementation order records the original progression. The current
repository includes the vertical slices through PR 8; the entries remain a
useful sequence for future extensions and stronger physical validation.

Each stack entry updates this page, adds focused correctness tests, keeps
expensive solves out of the quick test set, and documents the fields,
observables, multipliers, adapter, and output it introduces.

## Invariants for every stack entry

Moving from G0 to G1 should mainly change geometric evaluation or backend
machinery, not the mathematical `WeakTerm` or `Constraint` formulation.
Moving from P1 to P2 to P3 changes the coupling semantics, not necessarily
the geometry. Moving from G2 to G3 adds an observable lift or reconstruction,
not another coupling abstraction.

Finally, multiple active fields do not automatically imply a `Constraint`.
The mixed biharmonic formulation uses ordinary cross-equation weak terms;
constraints are reserved for relations whose multiplier represents an
auxiliary reaction.
