# Fiber-reinforced elastodynamics

This tutorial composes two independent, nonmatching elastodynamics Problems:

```text
matrix: ElastodynamicsSolver<dim,dim>
fiber:  ElastodynamicsSolver<1,dim>
```

The canonical input is
`tutorials/fiber_reinforced_elastodynamics/parameters.prm.in`:

```{literalinclude} ../../tutorials/fiber_reinforced_elastodynamics/parameters.prm.in
:language: ini
```

Run the 2D input with:

```bash
./build/fiber_reinforced_elastodynamics_debug \
  build/tutorials/fiber_reinforced_elastodynamics/parameters.prm
```

The matrix occupies `[-1,1]^2`. The fiber is an independently meshed line
inside it. Its ambient vector components still have dimension two, but its FE
support is one-dimensional. The fiber coefficients are an additive excess
contribution coupled to the matrix through a line multiplier. A vector Field
on the line selects all ambient components (`spacedim`), not just the
one-dimensional support component. Use `field.component_mask()` when passing
that selection to a native assembler.

For the fixed-step backward-Euler path, `matrix_only_displacement_difference()`
compares the coupled displacement with an uncoupled reference advanced from
its own displacement and velocity history under the same forcing and time
steps. It does not reuse the coupled state to construct the reference load.
This diagnostic is not an independent reference for the IDA path.

The application exposes matrix and fiber velocity fields and creates an
independent vector multiplier field. The constraint is assembled from
`weak_term(value(field), test(lambda))` terms. Its transpose reactions enforce
velocity compatibility on the two spaces. The five semantic fields are

```text
matrix.displacement  differential
matrix.velocity      differential
fiber.displacement   differential
fiber.velocity       differential
coupling.lambda      algebraic
```

The application registers these contributors with `IDAAdapter`, accepts the
displacement and velocity fields after accepted steps, and writes the two
Problem outputs plus the multiplier on its own FE space. The application
supports `2/2` and `3/3` matrix dimension selections; the fiber is
one-dimensional in both cases.
