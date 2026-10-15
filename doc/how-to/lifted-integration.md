# Integrate over a lifted geometry

`Domain::lift()` creates a geometric quadrature support from a representative
mesh and a `TensorProductLift` descriptor. It does not create a finite-element
space or a Field on the lifted geometry.

```cpp
Domain<1, 3> centerline(domain_parameters);
centerline.make_grid();

TensorProductLift<1, 2, 3> lift("/Tube geometry/");
lift.thickness = "0.02";
auto dV = centerline.lift(lift);
```

The support's `locally_owned_quadrature_points()` returns physical points,
their representative coordinates, physical weights, source cell and
quadrature indices, the local owner rank, and a rank-independent stable
identifier. The quadrature
uses a degree-2 deal.II mapping for the representative geometry and the same
cross-section transformation and weights as `TensorProductLiftSupport`.
For a lifted one-dimensional curve in three dimensions, normals are available
on locally straight representative cells. Curved cells retain their
quadrature, while `normal(index)` throws because this lift does not yet
differentiate the transported cross-section frame along the curve.

The support holds non-owning references to the `Domain` and lift descriptor;
both must outlive it. It generates points only on locally owned representative
cells, so a rank with no such cells has an empty point list. Call `dV.rebuild()`
after changing the mesh, lift parameters, or other geometry inputs. Thickness
can be a positive numeric constant or a state-independent spatial evaluator
passed to `lift(descriptor, evaluator)`. The evaluator receives only the
representative point, so it cannot depend on the current solver state or time.
Symbolic and active geometry dependencies are not supported.

The support exposes geometry for direct integration and for weak-term assembly
that evaluates its fields at physical points. The assembly path distributes
the points to a distributed bulk mesh with `ParticleCoupling`; trial and test
fields must share that bulk triangulation and mapping, and the lifted points
must lie in its locally searchable domain. Linear terms and nonlinear
observables with frozen finite-element inputs can use the same support.
Frozen inputs must share the source mesh and mapping.
No synthetic Field or DoFHandler is created on the lifted geometry. Trial and
test fields on separate bulk meshes are not supported. Shape derivatives of
the lift geometry are not included in the weak-term Jacobian.

```cpp
auto dV = centerline.lift(lift);
auto term = value(u_bulk) * test(v_bulk) * dV;
term.add(builder);
```

## Measure support-generation cost

Configure the optional benchmark target with
`-DIMMERSX_BUILD_BENCHMARKS=ON`, then run:

```bash
build-debug/benchmarks/lifted_integration_support_benchmark_debug \
  --refinements=4 --repetitions=9
```

The JSON result separates representative-mesh setup, support generation,
background-mesh/DoF setup, cold and warm bulk mass assembly, and matrix
application. Every timed MPI region starts after a barrier and reports the
maximum rank wall time. Assembly includes matrix materialization; application
uses the matrix just assembled. Numerical checks compare the global measure
and constant-field action with the straight-strip area. No equivalent native
Domain-lift or bulk-bulk trace assembler is available, so the native timing
ratio remains N/A. See `benchmarks/README.md` for the additive JSON fields.

Lifted quadrature is not split at background-cell interfaces. Refine the
representative mesh and cross-section to resolve geometry and piecewise FE
traces; increasing only the Gauss order need not give monotone convergence.
See [Choose where a weak term is integrated](support-aware-weak-terms) for
the independent-support convergence test and fixed-geometry limits.
