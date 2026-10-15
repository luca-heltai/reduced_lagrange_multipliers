# Weak term assembly benchmark

The `particle_penalty_benchmark` adds an independent explicit particle oracle
and the step-80 `DoFHandlerCoupling` penalty matrix reference, scalar and vector,
for 1-in-2, 2-in-3, 1-in-3, 2-in-2 and 3-in-3 nonmatching meshes. See
[`doc/performance/particle-penalty.md`](../doc/performance/particle-penalty.md)
for its correctness checks, timing scope, compatibility-header provenance and
performance graph integration. It runs with distributed background meshes and
matrices on one or multiple MPI ranks. CTest checks one and two ranks; the
performance graph records separate one/two-rank series using maximum-rank wall
times. This is an assembly comparison, not a scaling study.

Configure an out-of-source Debug build with the optional benchmark enabled:

```bash
cmake -S . -B build-bench-debug \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DCMAKE_BUILD_TYPE=Debug \
  -DIMMERSX_BUILD_BENCHMARKS=ON
cmake --build build-bench-debug --target weak_term_assembly_benchmark_debug -j
./build-bench-debug/benchmarks/weak_term_assembly_benchmark_debug \
  --refinements=4 --repetitions=9
```

The executable compares natural ImmersX mass and Poisson weak terms on explicit
cell supports with `dealii::MatrixCreator::create_mass_matrix` and
`create_laplace_matrix`. It emits JSON lines: Q1 volume mass and
Laplace matrices and constant volume source right-hand sides in 1D, 2D, and 3D;
boundary source right-hand sides and normal flux residuals are also compared
with `VectorTools` in 2D and 3D. One additional nonlinear case compares a
heterogeneous Q1/Q2 expression with a frozen DGQ0 input against an independent
`MeshWorker::mesh_loop` residual/Jacobian assembly. deal.II does not support
those boundary right-hand-side operations in 1D, so those cases are skipped.
Each pair uses
the same triangulation, DoFHandler, mapping, `QGauss(degree + 1)`, constraints,
and serial matrix representation. Each timed matrix assembly includes
construction of its sparsity pattern and matrix. Common mesh, finite element,
and DoF setup is reported separately. Matrix cases report cold timings and
alternate repeated
assemblies after an untimed warmup. The RHS cases report warmed repeated
assemblies. The normal flux case times ImmersX matrix assembly plus matrix
action, while `VectorTools` times RHS assembly, so those timings measure
different work. The JSON marks this case as non-equivalent and omits the timing
ratio. Timings include the median and median absolute deviation (MAD). A
relative Frobenius matrix or vector difference is checked before timing is
accepted.

Each JSON object is printed to standard output with the deal.II version, cell
and DoF counts, warm medians and MADs, and relative matrix or right-hand-side
error. Matrix cases also report common setup time and cold timings. Timings are
diagnostic and are not CI pass/fail thresholds. The matching cases cover
Q1 cell supports in 1D, 2D, and 3D; nonmatching and independent-support cases
are described below. Additional forms should use the same paired setup and
numerical comparison before their timings are reported.

The `heterogeneous_nonlinear_q1_q2_dgq0` case integrates
`(u * (1 + grad(w)[0]) + c) * test(v)` on cells selected by material id, with
Q1 `u` and `v`, Q2 `w`, and frozen DGQ0 `c`. Its MeshWorker reference assembles
the residual and both active-field Jacobians with deal.II cell traversal,
FEValues scratch data, affine-constraint scatter, and copy machinery. It reports
relative residual, matrix, and matrix-action errors before warmed timing
samples. Both paths return the residual and both Jacobians; MeshWorker fuses
their work in one cell pass while ImmersX uses separate residual and per-field
Jacobian calls. The ratio therefore compares the end-to-end public assembly
paths and records this traversal difference, rather than equal pass counts.

The installed deal.II package used for the baseline is `9.8.0-rc1`. Its
headers provide `MatrixCreator::create_mass_matrix`,
`MatrixCreator::create_laplace_matrix`, and
`MatrixCreator::create_boundary_mass_matrix` in
`deal.II/numerics/matrix_creator.h`; independent-mesh coupling assembly is in
`deal.II/non_matching/coupling.h`. The installed immersed-surface FEValues
header is named `deal.II/non_matching/fe_immersed_values.h` (it declares
`FEImmersedSurfaceValues`); there is no
`deal.II/non_matching/fe_immersed_surface_values.h` in this installation.
The CI workflow currently uses deal.II `9.7.1`, so APIs used by later cases
must also be checked against that version.

The benchmark does not initialize MPI. Its matching serial case does not need
MPI collectives, which also lets it run in environments where `MPI_Init_thread`
cannot initialize the MPI shared-memory transport. Distributed benchmarks must
initialize MPI, synchronize before timing, and report the maximum rank wall
time.

The `independent_background_trace_mass` and
`independent_rotated_trace_surface_3d` cases measure ImmersX assembly and
operator application for background-background Q1 trace mass matrices over an
independent immersed line and rotated planar surface. The line case reports
setup, cold assembly, warm assembly, and operator application separately. The
rotated surface case reports cold and warm assembly and operator application.
Both cases check the integrated measure against its analytic value; the line
case also checks the quadratic x moment. The installed deal.II API has no native
bulk-bulk trace-matrix assembler for this geometry, so these cases report no
native timing ratio. Their results are ImmersX-only timings, not comparisons
with `NonMatching::create_coupling_mass_matrix`, which assembles a different
rectangular operation.

All independent trace assembly timers include `matrix()` materialization.
The current explicit linear path assembles eagerly in `add()`; the explicit
materialization also keeps the timing correct if that path becomes deferred.
Warm application uses the newly assembled matrix, not the first matrix.
The serial benchmark does not initialize MPI and must not be launched with
multiple ranks. Use `lifted_integration_support_benchmark` for distributed
measurements; it synchronizes before each timed region and reduces elapsed
times with the MPI maximum.

### Additive JSON fields in the review follow-up

Existing case names and timing fields are preserved. The nonlinear record adds
`timing_scope: "end_to_end_residual_and_two_jacobians"`,
`immersx_assembly_calls: 3`, and `native_cell_traversals: 1`. These describe
equivalent outputs, not equivalent traversal counts.

The lifted record adds `deal_ii_version`, `background_setup_ms`,
`background_dofs`, `cold_operator_action_ms`, `operator_action_median_ms`,
`operator_action_mad_ms`, `analytical_action_error`, and
`timing_scope: "barrier_then_max_rank_wall_time"`. Its existing `setup_ms`
still means representative-mesh setup; `cold_support_build_ms` measures
support generation, and `first_bulk_assembly_ms` includes materialization.
Warm assembly constructs a fresh model and matrix each time; it does not
measure application of a retained operator. Native ratios remain N/A where
no equivalent native operation exists.

Consumers must tolerate absent optional fields in historical CP8 records.
Recorded Debug and Release reports and their provenance are in
`benchmarks/reports/`.
