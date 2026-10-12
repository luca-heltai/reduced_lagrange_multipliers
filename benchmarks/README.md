# Weak term assembly benchmark

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
`create_laplace_matrix`. It emits nine JSON lines: Q1 volume mass and Laplace
matrices, plus constant-source right-hand sides compared with
`VectorTools::create_right_hand_side`, in 1D, 2D, and 3D. Each pair uses the
same triangulation, DoFHandler, mapping, `QGauss(degree + 1)`, constraints, and
serial matrix representation. Each timed matrix assembly includes construction
of its sparsity pattern and matrix. Common mesh, finite element, and DoF setup
is reported separately. Matrix cases report cold timings and alternate repeated
assemblies after an untimed warmup. The RHS case reports warmed repeated
assemblies. Timings include the median and median absolute deviation (MAD). A
relative Frobenius matrix or vector difference is checked before timing is
accepted.

Each JSON object is printed to standard output with the deal.II version, cell
and DoF counts, warm medians and MADs, and relative matrix or right-hand-side
error. Matrix cases also report common setup time and cold timings. Timings are
diagnostic and are not CI pass/fail thresholds. The current benchmark covers
matching Q1 cell supports in 1D, 2D, and 3D. Additional forms should use the
same paired setup and numerical comparison before their timings are reported.

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
