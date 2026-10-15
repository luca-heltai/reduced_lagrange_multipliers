# Nonmatching particle penalty comparison

The `particle_penalty_*` cases compare the square background-space matrix

```{math}
A_{ij}=\gamma\sum_{q\in\Omega_2}\varphi_i(x_q)\varphi_j(x_q)JxW_q
```

with both basis functions on the background grid $\Omega_1$. For vector fields,
the product is a dot product; different components have zero cross-couplings.
This is the penalty part of step-80, not a complete Nitsche formulation and not
the rectangular immersed-to-background coupling mass matrix.

## Independent native oracle

The benchmark first generates `QGauss(3)` points and physical weights with
`FEValues` on the immersed mesh. It inserts particles manually in a deal.II
`Particles::ParticleHandler` on the background. Each owned integration cell
supplies its points and weights to `insert_global_particles`; deal.II searches
and transfers them to the background-cell owners. A separate loop evaluates
background shape functions at those locations and accumulates the matrix through
`AffineConstraints::distribute_local_to_global`. This code uses no ImmersX
assembly, Field, support or transfer APIs.

It then independently builds the same matrix with
`dealii::NonMatching::DoFHandlerCoupling::create_nitsche_restriction_matrix`, copied
from [deal.II PR #18888](https://github.com/dealii/dealii/pull/18888), pinned to
`c061c24bd41b4eb9fb62d1625dc0b818a54bc2bd`. The compatibility snapshot is in
`include/immersx/compatibility/dealii_9_9/dof_handler_coupling.h`. Its matrix
assembly loop is unchanged. Particle generation accepts serial meshes and
requests normals only in codimension one; penalty integration uses only `JxW`.
The adjacent `README.md` records these adaptations and licensing. This API is
from an open upstream PR, not a released deal.II 9.9 guarantee.

Only after the two native matrices agree does the benchmark construct ImmersX
objects. It materializes the matrix for:

```cpp
const auto space = ImmersX::fe_space(dof_handler, mapping, constraints,
                                     locally_relevant_dofs);
ImmersX::StateLayout layout;
const auto u = ImmersX::scalar_field(space, layout, "u");
const ImmersX::IntegrationSupport<1, 3> support(immersed_line);
ImmersX::weak_term(2.75 * ImmersX::value(u), ImmersX::test(u), support)
  .add(builder);
```

Use `ImmersX::vector_field` instead of `scalar_field` for a background
`FESystem(FE_Q(1), spacedim)`; the integration expression is unchanged.
The source and test share the same background mesh and mapping. The independent
integration mesh can have any positive dimension up to the background dimension.
Matching-mesh and rectangular coupling operations keep their existing paths.

## Cases and correctness

Every scalar case has a corresponding vector case, with `spacedim` components.
The background uses `parallel::distributed::Triangulation` (p4est). The
integration geometry uses `parallel::shared::Triangulation`, which also supports
1D meshes: its geometry is replicated but its cell ownership is partitioned.
Only owned integration cells generate points; matrices and vectors have
distributed row/DoF ownership. This is not an independent serial solve per rank.
The template dimensions are:

| Integration mesh | Background mesh | Case prefix |
| --- | --- | --- |
| `Triangulation<1,2>` | `Triangulation<2,2>` | `particle_penalty_1d_in_2d` |
| `Triangulation<2,3>` | `Triangulation<3,3>` | `particle_penalty_2d_in_3d` |
| `Triangulation<1,3>` | `Triangulation<3,3>` | `particle_penalty_1d_in_3d` |
| `Triangulation<2,2>` | `Triangulation<2,2>` | `particle_penalty_2d_in_2d` |
| `Triangulation<3,3>` | `Triangulation<3,3>` | `particle_penalty_3d_in_3d` |

The background has global refinement followed by local refinement and hanging
nodes. The integration mesh is tilted, offset, independently refined once and
entirely inside the background. Both paths use exactly the same quadrature
rule and penalty `2.75`. The global point count is checked to prevent lost or duplicated
integration points. MPI fixtures must have particles owned by at least two
background ranks. Nine repetitions and three background refinements are the
performance pipeline defaults; all ten cases run on one and two ranks.

Every owned matrix row is checked against all global columns, including absent
sparsity entries. Global sums/maxima give relative Frobenius difference and
maximum absolute entry difference for native versus manual and ImmersX versus
native. A nonconstant constrained-vector action is also compared globally.
A separate partition-of-unity check verifies that a constant vector integrates
to `penalty * physical_measure * components`, preventing replicated MPI
contributions from passing merely because both assembly paths duplicate them.
Both native paths call `compress(add)` after scatter; the upstream assembly
kernel itself does not compress the matrix. All error checks must be
strictly below `1e-12`; no bitwise-equality claim is made because summation and
point-search order can differ. A failed comparison stops the benchmark and the
performance harness archives the failed attempt rather than accepting timings.

[Issue #256](https://github.com/luca-heltai/immersx/issues/256) records the original
codimension-one restriction and rejection of independent full-dimensional
supports. These cases are the regression checks for the extended linear path.
CTest runs all ten geometries in both serial and two-rank MPI modes, with the
MPI checks labelled `integration;quick;mpi`. A two-rank correctness test is not
a scaling study; larger rank counts can be requested manually.

## Timing and performance graph

The timed native call constructs a fresh `DoFHandlerCoupling`, generates and
locates particles, initializes a matrix from the common background sparsity,
and assembles it. The timed ImmersX call registers the term, generates/locates
quadrature points, builds its sparsity and materializes the complete matrix.
Mesh/DoF setup and correctness checks are outside both timed regions. A warmup
precedes alternating native/ImmersX calls. Each sample starts after an MPI
barrier and records the maximum elapsed wall time across ranks; only rank zero
writes JSONL. These are equivalent final operators
but different public assembly paths: the ratio includes ImmersX sparsity setup
and the native helper's particle metadata construction. It does not compare
cached kernels or claim equal internal work.

The [performance history](index) exposes absolute medians/MAD, the
ImmersX/native ratio, and compatible historical changes for each case. Filter
`case` by its full `_scalar` or `_vector` name and select the corresponding
`warm_immersx` or `warm_dealii` chart series. Select `ranks=1` or `ranks=2`;
rank counts form separate comparison series. Harness version 4 isolates this
expanded MPI suite; the benchmark, manual oracle and coupling-header Git blobs are
recorded and included in compatibility keys. Old reports are not re-labelled
as measurements of these new cases. Raw JSONL and numerical checks remain
available in the attempt details.

```bash
cmake -S . -B /tmp/immersx-penalty-build \
  -DCMAKE_BUILD_TYPE=Release -DIMMERSX_BUILD_BENCHMARKS=ON
cmake --build /tmp/immersx-penalty-build --target particle_penalty_benchmark -j2
/tmp/immersx-penalty-build/benchmarks/particle_penalty_benchmark \
  --refinements=2 --repetitions=9
mpirun -np 2 /tmp/immersx-penalty-build/benchmarks/particle_penalty_benchmark \
  --refinements=2 --repetitions=9
ctest --test-dir /tmp/immersx-penalty-build -R 'ImmersX.ParticlePenalty' \
  --output-on-failure
```

Debug uses the `_debug` executable suffix. The reusable performance workflow
builds and runs this executable automatically before generating documentation.
Historical SHAs predating the executable cannot supply this measurement; manual
backfill preserves that failure/missing status rather than running new code
under an old SHA.
