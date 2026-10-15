# Weak-term benchmark report

## Historical review follow-up measurements

The `*-review-*.jsonl`, `weakterm-review-manifest.json`, and
`weakterm-review-validation.json` files are retained byte for byte from the
original review follow-up. The measured source revision is
`1e00b166f1471453e196fe91af34e5af1453701d`, not the repaired head or the
later commit that adds the reports. The manifest records its source tree,
artifact checksums, commands, compiler and host configuration.

These six JSONL files contain 40 records: 18 serial cases in each mode and
one lifted case at one and two ranks in each mode. They use 40 repetitions,
three serial refinements and four lifted refinements. Their timings are
historical diagnostic measurements, not performance gates or a scaling study.
Validation claims in the retained JSON refer to that original source revision;
they are not new validation of the repaired stack.

## Historical CP8 measurements

These are historical reports retained from the original stack, not measurements
of the repaired head. Their files do not record an immutable source SHA; do not
assign them a new SHA. Fresh measurements must record their source SHA.

These JSONL files record the full serial weak-term benchmark on macOS arm64
with deal.II 9.8.0-rc1. Both use 3 global refinements and 40 timed
repetitions:

- `weakterm-cp8-debug-dealii-9.8.0-rc1-arm64.jsonl` was built with
  `CMAKE_BUILD_TYPE=Debug`.
- `weakterm-cp8-release-dealii-9.8.0-rc1-arm64.jsonl` was built with
  `CMAKE_BUILD_TYPE=Release`.

Run `weak_term_assembly_benchmark --refinements=3 --repetitions=40` from a
Debug or Release build with `IMMERSX_BUILD_BENCHMARKS=ON` to repeat the cases.
Timings on a repaired head are separate measurements.
Records include numerical errors and warm medians/MADs. Mesh, quadrature,
setup, cold and application fields vary by case; an absent field does not mean
zero time. Native ratios are reported only for comparable outputs.

The `boundary_flux_q1` result records absolute timings and errors but has no
ratio: ImmersX evaluates the weak term through its prepared matrix path, while
deal.II assembles the boundary right-hand side directly. The independent
background-trace and rotated-surface cases also have no deal.II ratio because
deal.II provides no equivalent bulk-bulk trace assembler; their records report
absolute timings and analytical measure/action errors.

For the 3D Q1 mass case (729 DoFs, 512 cells), the Debug warm ratio changed from
3.95 before to 2.44 after reusing each trial and test basis value across the
local matrix entries at a quadrature point. The matching ImmersX median changed
from 11.47 ms to 7.31 ms; the native median was 2.90 ms in the before run and
3.00 ms in the after run. Both runs used 40 repetitions and reported zero
relative matrix error. The historical Release run reports a warm ratio of 2.10
for the same case.

A macOS `sample` profile of the previous Debug benchmark placed the 3D mass
work in `WeakAssembly::assemble_from_source()` and showed repeated
`FEValuesViews::Scalar::value()` calls. The sample also included dynamic-loader
startup and is diagnostic rather than a precise phase breakdown. The
per-quadrature basis-value reuse targets the repeated reads visible in that
profile.

At collection time, the lifted-support benchmark attempted matrix
materialization inside its timed region but aborted during `MPI_Init_thread`
with `MPIDI_POSIX_comm_bootstrap: Out of memory`, before producing timings.
This is a historical limitation, not the status of current correctness tests.

`weakterm-cp5-nonlinear-q1-q2-dgq0-dealii-9.8.0-rc1-arm64.jsonl` records the
40-repetition Debug and Release timings for the heterogeneous nonlinear
Q1/Q2/DGQ0 case before and after reusing input FEValues across cells with a
shared mapping. After the change, the relative residual error is `1.37e-16`,
and the Q1 Jacobian, Q2 Jacobian, and Jacobian-action errors are zero for this
case. The warm ImmersX to MeshWorker ratio changed from 4449 to 8.10 in Debug
and from 5212 to 2.59 in Release. MeshWorker assembles the residual and both
Jacobians in one cell pass; ImmersX measures its three public assembly calls,
so these ratios include the different traversal counts.
