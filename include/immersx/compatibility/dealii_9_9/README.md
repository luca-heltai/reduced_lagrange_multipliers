# deal.II 9.9 compatibility: DoFHandlerCoupling

`dof_handler_coupling.h` is copied from
[deal.II PR #18888](https://github.com/dealii/dealii/pull/18888), commit
`c061c24bd41b4eb9fb62d1625dc0b818a54bc2bd` (`step-80-new`), file
`include/deal.II/non_matching/dof_handler_coupling.h`. The upstream SPDX license
and copyright notice are retained. This is a snapshot of an open upstream PR,
not an assertion that its API has shipped in deal.II 9.9.

The `create_nitsche_restriction_matrix` assembly loop is unchanged. It sums
`penalty * shape_value(i, reference_q) * shape_value(j, reference_q) * JxW`
only for matching FE components and scatters through `AffineConstraints`.
The benchmark compares it against a separately written explicit particle loop
before using it as the native oracle for ImmersX.

Local backport adjustments:

- Include dependencies used by the header explicitly (FEValues, FullMatrix,
  distributed triangulation and ObserverPointer).
- In the quadrature-particle generator, use the base triangulations rather
  than requiring two `parallel::distributed::Triangulation` casts. Count locally
  owned cells by traversal. This supports serial immersed lines, for which
  p4est triangulations are not available, without changing the weight or basis
  evaluation used by the penalty matrix.
- Request and evaluate normals only for codimension one. Codimension-two
  particle properties retain zero normal placeholders; the penalty matrix
  uses only the physical quadrature weight, not normals.
- Apply the repository's C++ formatter.

The other upstream coupling methods are retained but are not advertised as
verified backports. In particular, interpolation retains upstream distributed
mesh assumptions. The tests exercise only the penalty matrix and quadrature
particle access with primitive scalar/vector Q1 elements, including multiple
MPI ranks. The background is p4est-distributed; integration cells are partitioned
by `parallel::shared::Triangulation`, including 1D geometry. The caller must
`compress(VectorOperation::add)` after the upstream matrix assembly loop; the
kernel itself does not perform that collective.

The declaration is in `dealii::NonMatching`, with the upstream include guard.
Do not include a different version of `DoFHandlerCoupling` in the same translation
unit. Installed applications should choose either this pinned compatibility
header or the version supplied by their deal.II installation, not both.
