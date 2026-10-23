# Coupled Poisson–elasticity

This application composes a 1D-in-3D Poisson Problem with a 3D static
elasticity Problem. The Poisson solution is multiplied by two, lifted from
the representative cylinder, converted to a normal traction, and assembled
against the elasticity displacement test field:

```text
PoissonSolver<1,3> -> value -> lift -> normal traction
                                      -> test(displacement)
ElasticStaticProblem<3,3> ------------^
```

The executable is `coupled_poisson_elasticity`, from
`apps/app_coupled_poisson_elasticity.cc`. Its input is
`tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in`:

```{literalinclude} ../../tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in
:language: ini
```

The application calls `setup()` on each Problem and adds both Problems to a
`LinearAdapter`. It forms the traction from their semantic Fields with
`make_lift(2. * value(poisson.solution), pressure_lift) * normal(surface)` and
adds the resulting `weak_term` against `test(elastic.displacement)`. The adapter
owns the execution vector and coupled solve. `OutputHandler` writes both
semantic Fields directly from the adapter state.

Run the configured input with:

```bash
mpirun -np 1 ./build/coupled_poisson_elasticity_debug \
  build/tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm
```

The executable writes the pressure and displacement Fields below the configured
output directory. Integration tests check the coupled residual, pressure scaling,
and traction balance; the application test checks the configured output.
