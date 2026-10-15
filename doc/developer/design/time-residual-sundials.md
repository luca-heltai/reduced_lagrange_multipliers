# Residual execution with IDA and KINSOL

The semantic execution model exposes a residual and its two state
derivatives. A contributor adds residual operations to the shared
`SemiDiscreteModel` and registers separate operators for
`dF/dy` and `dF/dydot`. The execution adapter evaluates these operations for
the supplied state; the evaluation context and its state views are not stored
after the evaluation.

## IDA

`IDAAdapter` connects the semantic model to deal.II's SUNDIALS IDA wrapper.
The application registers contributors before solving, creates a state and a
state derivative, and supplies accepted-state output when native data must be
updated:

```cpp
TimeIntervalParameters time_interval;
IDAParameters ida_parameters;
IDAAdapter<FieldVector, GlobalVector> ida(time_interval,
                                         ida_parameters,
                                         MPI_COMM_WORLD);
auto fields = ida.add(problem, "solid");

ida.set_output_step(
  [&problem, &ida, fields](double time,
                           const GlobalVector &state,
                           const GlobalVector &state_dot,
                           unsigned int step) {
    problem.accept_state(
      ida.field(state, fields.fields().displacement),
      ida.field(state, fields.fields().velocity),
      time,
      step);
    (void)state_dot;
  });

auto state = ida.make_state();
auto state_dot = ida.make_state();
ida.solve(state, state_dot);
```

The application fills the initial state and derivative before `solve()`. IDA
receives a differential-component mask built from each field descriptor. A
field can be fully differential, fully algebraic, or mixed by component.

IDA asks for the solver Jacobian

```{math}
J = \frac{\partial F}{\partial y}
  + \alpha\frac{\partial F}{\partial \dot y}.
```

`alpha` belongs to IDA and is applied inside `IDAAdapter`. Contributors and
Problems provide the two derivative operators separately. The adapter can
use its built-in GMRES/FGMRES path or an application-supplied linear solve
callback.

`TimeIntervalParameters` owns the initial and final times and the physical
output interval. `IDAParameters` owns IDA's adaptive-step controls, error
tolerances, differential/algebraic error handling, and initial-condition
correction settings. The adapter combines the two into deal.II's IDA
`AdditionalData`; the generic fixed-step controls are not used by IDA.

The adapter uses the pinned compatibility implementation of
[deal.II PR #20260](https://github.com/dealii/dealii/pull/20260), not the
installed IDA wrapper. Its type is
`ImmersX::Compatibility::SUNDIALS::IDA<GlobalVector>`. Existing parameter-file
settings and defaults are unchanged. To set the additional upstream controls,
copy the assembled configuration before calling `solver()` or `solve()`:

```cpp
auto controls = ida.additional_data();
controls.maximum_step_size = 0.01;
controls.maximum_number_of_steps = 2000;
controls.linear_tolerance_factor = 0.05;
ida.set_additional_data(controls);
```

The compatibility `AdditionalData::add_parameters()` exposes all upstream
controls when used directly with a `ParameterHandler`; the adapter's
`IDAParameters` continues to expose its existing parameter-file subset.
Adaptivity controls require SUNDIALS 6.2 or newer; the constraint-failure limit
requires 7.6 or newer.

After registering contributors, `ida.solver()` exposes the constraint and
expert setup callbacks. For inequality constraints, create an execution vector
with `ida.make_state()`, fill it through `ida.field()`, and return it from
`ida.solver().get_constraint_vector`. Entries encode unconstrained (0),
nonnegative/nonpositive (+1/-1), or strictly positive/negative (+2/-2) values.
Keep the vector alive through the solve and resize it after any layout change.
An all-zero constraint vector means no constraints.

`ida.solver().custom_setup` runs on every reset, after the wrapper installs its
callbacks and before initial-condition correction. `get_ida_memory()` permits
IDA queries; it is null before the first reset. Both that pointer and
`get_sun_context()` (SUNDIALS 6+) are invalidated by reset. Do not replace
wrapper-owned user data or callbacks.

The optional
`set_compute_consistent_initial_conditions(time, state, state_dot)` callback
can update both vectors when the correction type is `use_y_diff`. The optional
`set_solver_should_restart()` callback can transfer state after an application
change and request an IDA restart. These callbacks operate on execution
vectors; updating Problem-owned matrices, constraints, and accepted state is
the application's responsibility.

## KINSOL

`KINSOLAdapter` connects the same semantic contributor model to deal.II's
SUNDIALS KINSOL wrapper for a steady nonlinear problem:

```cpp
KINSOLAdapterParameters parameters;
KINSOLAdapter<FieldVector, GlobalVector> kinsol(parameters, MPI_COMM_WORLD);
auto fields = kinsol.add(problem, "steady");
auto state = kinsol.make_state();
kinsol.solve(state);
```

KINSOL requires `F(y)=0`; contributors with derivative terms are rejected.
The adapter prepares the semantic Jacobian for each nonlinear solve and uses
either the configured linear solve callback or its built-in GMRES path.
`KINSOLAdapterParameters` selects `newton` or `linesearch` and sets the
nonlinear stopping and Jacobian-setup controls.

Both adapters use deal.II `LinearOperator` objects for Jacobian actions. A
native matrix can be exposed through `SemidiscreteBuilder::matrix_operator()`;
the Problem that owns the matrix must outlive the contributor.
