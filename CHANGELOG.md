# Changelog

All notable changes to ImmersX are documented in this file. Each entry
begins with a link to the pull request that introduced the change.

The file is included in the documentation and is published at
`doc/changes.md`. Add new entries under `[Unreleased]` using the
skeleton in `.github/CHANGELOG_TEMPLATE.md`.

## [Unreleased]

### Added

- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Added explicit embedded-mesh integration supports and native deal.II nonmatching mass assembly for value pairings.
- [PR #242](https://github.com/luca-heltai/immersx/pull/242) Added boundary integration supports and normal, conormal, and tangential expressions for linear weak terms.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Added cell supports for selecting the integration domain of linear weak terms.
- [PR #240](https://github.com/luca-heltai/immersx/pull/240) Added an opt-in benchmark for comparing matching mass assembly with deal.II.
- [PR #236](https://github.com/luca-heltai/immersx/pull/236) Add an explicit forward-Euler elasticity solver with a lumped mass matrix.
- [PR #235](https://github.com/luca-heltai/immersx/pull/235) Add a frequency-domain elasticity solver.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Add Coral operations for domain triangulations, VTK output, and output handlers.
- [PR #231](https://github.com/luca-heltai/immersx/pull/231) Added primitive static elasticity boundary conditions.
- [PR #229](https://github.com/luca-heltai/immersx/pull/229) Added Poisson primitive right-hand-side terms.
- [PR #228](https://github.com/luca-heltai/immersx/pull/228) Exposed compositional Coral boundary conditions.
- [PR #227](https://github.com/luca-heltai/immersx/pull/227) Exposed compositional Coral finite-element expressions.
- [PR #225](https://github.com/luca-heltai/immersx/pull/225) Added time-aware aggregable boundary conditions.
- [PR #224](https://github.com/luca-heltai/immersx/pull/224) Added solver-neutral known terms and boundary integration.
- [PR #220](https://github.com/luca-heltai/immersx/pull/220) Added the generic primitive scalar Poisson path.
- [PR #217](https://github.com/luca-heltai/immersx/pull/217) Added the solver-neutral `OutputHandler`.
- [PR #195](https://github.com/luca-heltai/immersx/pull/195) Added the KINSOL steady nonlinear adapter.
- [PR #193](https://github.com/luca-heltai/immersx/pull/193) Added nonlinear Observables and the MetricFlowX radial law.
- [PR #190](https://github.com/luca-heltai/immersx/pull/190) Implemented the tensor-product lifting backend and cleaned up transposes.
- [PR #184](https://github.com/luca-heltai/immersx/pull/184) Composed unified constraint sums.
- [PR #183](https://github.com/luca-heltai/immersx/pull/183) Supported scaled weak-term observables.
- [PR #178](https://github.com/luca-heltai/immersx/pull/178) Added the cached nonmatching weak-term backend.
- [PR #175](https://github.com/luca-heltai/immersx/pull/175) Added the finite-element space and Observable foundation API.
- [PR #174](https://github.com/luca-heltai/immersx/pull/174) Restored the standalone Poisson and ElasticStatic execution paths.
- [PR #173](https://github.com/luca-heltai/immersx/pull/173) Supported mixed-dimensional elastodynamics execution paths.
- [PR #168](https://github.com/luca-heltai/immersx/pull/168) Completed two-way MetricFlowX vessel-wall coupling.
- [PR #167](https://github.com/luca-heltai/immersx/pull/167) Added one-way MetricFlowX-Elastodynamics coupling.
- [PR #164](https://github.com/luca-heltai/immersx/pull/164) Added application-consistent initial conditions to the IDA adapter.
- [PR #161](https://github.com/luca-heltai/immersx/pull/161) Added optional MetricFlowX integration.
- [PR #132](https://github.com/luca-heltai/immersx/pull/132) Supported active and frozen finite-element expression sources.
- [PR #131](https://github.com/luca-heltai/immersx/pull/131) Added reusable imported finite-element fields.
- [PR #130](https://github.com/luca-heltai/immersx/pull/130) Coupled Poisson gradients and imported path fields to elasticity.
- [PR #125](https://github.com/luca-heltai/immersx/pull/125) Composed expression representations with the tensor-product lift.
- [PR #124](https://github.com/luca-heltai/immersx/pull/124) Inferred gradient sampling requirements for expressions.
- [PR #122](https://github.com/luca-heltai/immersx/pull/122) Added the scalar expression representation.
- [PR #121](https://github.com/luca-heltai/immersx/pull/121) Added retained finite-element sampling operators.
- [PR #120](https://github.com/luca-heltai/immersx/pull/120) Added the symbolic expression kernel.
- [PR #118](https://github.com/luca-heltai/immersx/pull/118) Added the parameterized tensor-product lift.
- [PR #116](https://github.com/luca-heltai/immersx/pull/116) Productionized the parameterized ElasticStatic problem.
- [PR #115](https://github.com/luca-heltai/immersx/pull/115) Added the coupled Poisson-to-elasticity vertical slice.
- [PR #107](https://github.com/luca-heltai/immersx/pull/107) Added geometry-aware Representation lifting.
- [PR #106](https://github.com/luca-heltai/immersx/pull/106) Added the weak `CouplingOperator` boundary.
- [PR #103](https://github.com/luca-heltai/immersx/pull/103) Added the standalone `ElasticStaticProblem`.
- [PR #102](https://github.com/luca-heltai/immersx/pull/102) Added the representation-driven load interaction.
- [PR #99](https://github.com/luca-heltai/immersx/pull/99) Added mixed Field component views.
- [PR #98](https://github.com/luca-heltai/immersx/pull/98) Added the minimal Field representation concept.
- [PR #94](https://github.com/luca-heltai/immersx/pull/94) Added deal.II semidiscrete contributors and automatic IDA composition.
- [PR #92](https://github.com/luca-heltai/immersx/pull/92) Added composable semidiscrete physics adapters.
- [PR #91](https://github.com/luca-heltai/immersx/pull/91) Added full-order fiber-reinforced elastodynamics.
- [PR #86](https://github.com/luca-heltai/immersx/pull/86) Added distributed IDA adapters for the standalone PDEs.
- [PR #85](https://github.com/luca-heltai/immersx/pull/85) Added the standalone transient Navier-Stokes solver.
- [PR #78](https://github.com/luca-heltai/immersx/pull/78) Added the representation-driven Poisson coupling prototype.
- [PR #75](https://github.com/luca-heltai/immersx/pull/75) Added the dimension-generic Poisson and Laplace-Beltrami solver.
- [PR #71](https://github.com/luca-heltai/immersx/pull/71) Allowed a zero reduced dimension in the tensor-product space.
- [PR #70](https://github.com/luca-heltai/immersx/pull/70) Made imported VTK fields usable in expressions.
- [PR #69](https://github.com/luca-heltai/immersx/pull/69) Added the VTK-backed reduced-coupling path to elasticity.
- [PR #62](https://github.com/luca-heltai/immersx/pull/62) Supported strong dynamic elasticity constraints.

### Changed

- [PR #237](https://github.com/luca-heltai/immersx/pull/237) Replace custom MetricFlowX coupling with structural finite-element observables and cell integration.
- [PR #226](https://github.com/luca-heltai/immersx/pull/226) Associated boundary conditions with semantic fields.
- [PR #222](https://github.com/luca-heltai/immersx/pull/222) Returned typed semantic fields from the contributors.
- [PR #215](https://github.com/luca-heltai/immersx/pull/215) Replaced the 2D Coral facade with generic composition.
- [PR #214](https://github.com/luca-heltai/immersx/pull/214) Unified the Coral parameter-loading nodes.
- [PR #213](https://github.com/luca-heltai/immersx/pull/213) Introduced `Domain`-based mesh parameters and migrated the solvers.
- [PR #204](https://github.com/luca-heltai/immersx/pull/204) Refactored time parameter and integrator ownership.
- [PR #200](https://github.com/luca-heltai/immersx/pull/200) Added the deliberate application-coupling roadmap.
- [PR #199](https://github.com/luca-heltai/immersx/pull/199) Reduced memory use in elasticity.
- [PR #189](https://github.com/luca-heltai/immersx/pull/189) Used typed deal.II finite-element expressions for weak terms.
- [PR #188](https://github.com/luca-heltai/immersx/pull/188) Completed the unified fiber-constraint follow-up.
- [PR #187](https://github.com/luca-heltai/immersx/pull/187) Migrated the fiber driver to unified weak-term constraints.
- [PR #185](https://github.com/luca-heltai/immersx/pull/185) Moved the normal pressure load into `weak_term`.
- [PR #181](https://github.com/luca-heltai/immersx/pull/181) Unified Lagrange multiplier constraints over weak terms.
- [PR #176](https://github.com/luca-heltai/immersx/pull/176) Optimized weak-term assembly for simple finite-element paths.
- [PR #166](https://github.com/luca-heltai/immersx/pull/166) Refactored time parameter ownership around `TimeParameters`.
- [PR #162](https://github.com/luca-heltai/immersx/pull/162) Added parameter objects to the execution adapters.
- [PR #160](https://github.com/luca-heltai/immersx/pull/160) Modernized applications around the semantic execution adapters.
- [PR #159](https://github.com/luca-heltai/immersx/pull/159) Gave multiplier Interactions native output ownership.
- [PR #158](https://github.com/luca-heltai/immersx/pull/158) Wrote native output after composed solves.
- [PR #157](https://github.com/luca-heltai/immersx/pull/157) Simplified linear operator composition.
- [PR #152](https://github.com/luca-heltai/immersx/pull/152) Repaired the final semidiscrete composition stack.
- [PR #138](https://github.com/luca-heltai/immersx/pull/138) Hardened field transfer and pruned stale reduced-field APIs.
- [PR #128](https://github.com/luca-heltai/immersx/pull/128) Built pressure from deferred finite-element expressions.
- [PR #123](https://github.com/luca-heltai/immersx/pull/123) Supported semantic expression dependencies.
- [PR #117](https://github.com/luca-heltai/immersx/pull/117) Brought `ElasticStatic` to parity with elasticity.
- [PR #114](https://github.com/luca-heltai/immersx/pull/114) Simplified the public multiphysics composition API.
- [PR #113](https://github.com/luca-heltai/immersx/pull/113) Clarified Representation quantity spaces.
- [PR #109](https://github.com/luca-heltai/immersx/pull/109) Consolidated the Representation domain, evaluation, and transfer boundaries.
- [PR #108](https://github.com/luca-heltai/immersx/pull/108) Separated geometry maps from Representation lifting.
- [PR #105](https://github.com/luca-heltai/immersx/pull/105) Separated the Representation source and evaluation domain.
- [PR #104](https://github.com/luca-heltai/immersx/pull/104) Clarified Representation evaluation and linearization.
- [PR #100](https://github.com/luca-heltai/immersx/pull/100) Made `TensorProduct` a representation-first lifting.
- [PR #97](https://github.com/luca-heltai/immersx/pull/97) Replaced `TimeRole` with per-field differential component masks.
- [PR #83](https://github.com/luca-heltai/immersx/pull/83) Unified semantic residual fields and added mixed DAE integration.
- [PR #81](https://github.com/luca-heltai/immersx/pull/81) Dispatched applications from parameter dimensions.
- [PR #79](https://github.com/luca-heltai/immersx/pull/79) Stabilized the representation and constraint interaction architecture.
- [PR #76](https://github.com/luca-heltai/immersx/pull/76) Separated the reduced-space core from the VTK backend.
- [PR #73](https://github.com/luca-heltai/immersx/pull/73) Improved distributed tensor-product coupling.

### Fixed

- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Use particle-based assembly for mixed-dimensional weak terms with a distributed background mesh.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Keep persisted Coral weak-term and continuity graphs readable across compilers.
- [PR #238](https://github.com/luca-heltai/immersx/pull/238) Avoid indexed writes to ghosted Trilinos vectors in `OutputHandler`.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Write the short Coral parameter values used by each graph run to a sibling file.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read point and cell fields from XML VTK mesh files.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Treat grid names with a file extension as mesh files even when the path is unresolved.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read XML VTK mesh files with VTK versions before 9.3.
- [PR #230](https://github.com/luca-heltai/immersx/pull/230) Fixed Coral first-run parameter initialization.
- [PR #223](https://github.com/luca-heltai/immersx/pull/223) Fixed the Coral CTest filter escaping.
- [PR #210](https://github.com/luca-heltai/immersx/pull/210) Fixed the Schur preconditioner breakdown on lucky termination.
- [PR #208](https://github.com/luca-heltai/immersx/pull/208) Fixed elastodynamics refinement and moving-boundary constraints.
- [PR #207](https://github.com/luca-heltai/immersx/pull/207) Fixed Newmark inhomogeneous boundary handling.
- [PR #205](https://github.com/luca-heltai/immersx/pull/205) Fixed the dynamic elasticity constraint instability.
- [PR #202](https://github.com/luca-heltai/immersx/pull/202) Fixed elastodynamics adapter constrained degrees of freedom.
- [PR #198](https://github.com/luca-heltai/immersx/pull/198) Fixed the inclusion catalog scalar read.
- [PR #191](https://github.com/luca-heltai/immersx/pull/191) Fixed sparse weak-term regression comparisons.
- [PR #180](https://github.com/luca-heltai/immersx/pull/180) Fixed the strict API documentation build.
- [PR #172](https://github.com/luca-heltai/immersx/pull/172) Fixed rotation-aware tensor-product vector modes.
- [PR #163](https://github.com/luca-heltai/immersx/pull/163) Fixed MetricFlowX IDA parameter construction.
- [PR #156](https://github.com/luca-heltai/immersx/pull/156) Fixed distributed Trilinos Schur preconditioner maps.
- [PR #155](https://github.com/luca-heltai/immersx/pull/155) Fixed the Fiber IDA initial solver lifetime.
- [PR #137](https://github.com/luca-heltai/immersx/pull/137) Fixed reduced-coupling assembly and deduplicated imported-field transfer.
- [PR #136](https://github.com/luca-heltai/immersx/pull/136) Fixed elasticity compilation without VTK.
- [PR #135](https://github.com/luca-heltai/immersx/pull/135) Fixed duplicate symbolic outputs in the tensor-product right-hand side.
- [PR #112](https://github.com/luca-heltai/immersx/pull/112) Fixed the matrix-free transfer test build.
- [PR #89](https://github.com/luca-heltai/immersx/pull/89) Fixed parallel elasticity assembly for the coupled case.
- [PR #74](https://github.com/luca-heltai/immersx/pull/74) Addressed a failing test.
- [PR #68](https://github.com/luca-heltai/immersx/pull/68) Removed compiler warnings with deal.II 9.8.
- [PR #67](https://github.com/luca-heltai/immersx/pull/67) Fixed the inclusion basis-function scaling.

### Removed

- [PR #192](https://github.com/luca-heltai/immersx/pull/192) Purged the transitional representation architecture.
- [PR #186](https://github.com/luca-heltai/immersx/pull/186) Removed the obsolete representation load interaction.
- [PR #96](https://github.com/luca-heltai/immersx/pull/96) Removed the semidiscrete compatibility layers.
- [PR #93](https://github.com/luca-heltai/immersx/pull/93) Removed the obsolete monolithic compatibility layer.

### Documentation

- [PR #248](https://github.com/luca-heltai/immersx/pull/248) Document reverse-chronological changelog ordering and checks for missing entries from merged PRs.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Allow pull requests without a related issue to omit the closing reference.
- [PR #196](https://github.com/luca-heltai/immersx/pull/196) Refreshed the documentation for the current architecture.
- [PR #95](https://github.com/luca-heltai/immersx/pull/95) Updated the agent instructions and the contributing guide.
- [PR #82](https://github.com/luca-heltai/immersx/pull/82) Revised the core architecture documentation.
- [PR #77](https://github.com/luca-heltai/immersx/pull/77) Documented the ImmersX core architecture.

### Tutorial

- [PR #234](https://github.com/luca-heltai/immersx/pull/234) Run the tutorial platform through noVNC with remote Slurm execution and selectable Debug or Release packages.
- [PR #233](https://github.com/luca-heltai/immersx/pull/233) Build multi-architecture Coral, visualizer, and platform images for the dealiiX tutorial stack.
- [PR #127](https://github.com/luca-heltai/immersx/pull/127) Updated the tutorial documentation.
- [PR #84](https://github.com/luca-heltai/immersx/pull/84) Added the standalone first-order elastodynamics solver and its tutorials.
- [PR #80](https://github.com/luca-heltai/immersx/pull/80) Added the random-particle ReducedPoisson tutorials.

### Build and packaging

- [PR #219](https://github.com/luca-heltai/immersx/pull/219) Added the Coral Debug and Release CI jobs.
- [PR #211](https://github.com/luca-heltai/immersx/pull/211) Configured the shared ccache and VS Code path mapping.
- [PR #201](https://github.com/luca-heltai/immersx/pull/201) Restored DebugRelease single-config builds.
- [PR #194](https://github.com/luca-heltai/immersx/pull/194) Cancelled superseded GitHub Actions runs.
- [PR #88](https://github.com/luca-heltai/immersx/pull/88) Prepared ImmersX for packaging.
- [PR #87](https://github.com/luca-heltai/immersx/pull/87) Centralized build-tree test inputs and added the contributing guide.

### Testing and CI

- [PR #248](https://github.com/luca-heltai/immersx/pull/248) Run pull-request tests for stacked branches targeting any branch.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Build and test Coral plugins against the `dealiix-tutorial` Coral branch.
- [PR #221](https://github.com/luca-heltai/immersx/pull/221) Added the issue #209 elastodynamics equivalence regression.
- [PR #182](https://github.com/luca-heltai/immersx/pull/182) Migrated the coupled regressions to unified constraints.
- [PR #177](https://github.com/luca-heltai/immersx/pull/177) Reorganized the tests into labeled CTest suites.
- [PR #169](https://github.com/luca-heltai/immersx/pull/169) Added the one-vessel two-way MetricFlowX-Elastodynamics MMS verification.
- [PR #90](https://github.com/luca-heltai/immersx/pull/90) Reduced the Debug testsuite runtime.

## [0.3] - 2026-05-11

### Added

- [PR #61](https://github.com/luca-heltai/immersx/pull/61) Added per-boundary-id Dirichlet and Neumann data with modulated parsed functions.
- [PR #60](https://github.com/luca-heltai/immersx/pull/60) Added fully distributed triangulation support to the elasticity problem.
- [PR #51](https://github.com/luca-heltai/immersx/pull/51) Added per-material-id material properties.

### Changed

- [PR #55](https://github.com/luca-heltai/immersx/pull/55) Grouped inclusions that fall in the same cell.
- [PR #54](https://github.com/luca-heltai/immersx/pull/54) Moved vessel-pressure computation and output into coupled elasticity.
- [PR #53](https://github.com/luca-heltai/immersx/pull/53) Cleaned up and refactored coupled elasticity.
- [PR #50](https://github.com/luca-heltai/immersx/pull/50) Incorporated the brain-dynamics changes contributed by Yashu.

### Fixed

- [PR #58](https://github.com/luca-heltai/immersx/pull/58) Fixed the reduced Poisson problem.

### Documentation

- [PR #64](https://github.com/luca-heltai/immersx/pull/64) Rebranded the project with a new logo and README.
- [PR #56](https://github.com/luca-heltai/immersx/pull/56) Full documentation pass over the missing classes.

### Build and packaging

- [PR #57](https://github.com/luca-heltai/immersx/pull/57) Updated the GitHub Pages workflow.

### Testing and CI

- [PR #59](https://github.com/luca-heltai/immersx/pull/59) Added the liver-benchmark mesh preparation files.

## [0.2] - 2026-01-29

### Added

- [PR #44](https://github.com/luca-heltai/immersx/pull/44) Added a `properties` member and a vertex- and cell-agnostic field transfer.
- [PR #42](https://github.com/luca-heltai/immersx/pull/42) Added rigid-body mode extraction for AMG preconditioning.
- [PR #40](https://github.com/luca-heltai/immersx/pull/40) Added VTK utilities for reading cell and point data.
- [PR #36](https://github.com/luca-heltai/immersx/pull/36) Added local refinement support for deal.II inclusions.
- [PR #32](https://github.com/luca-heltai/immersx/pull/32) Added initial deal.II-based inclusion support.
- [PR #31](https://github.com/luca-heltai/immersx/pull/31) Added parameter-file-driven inclusion data and corrected pressure and stress output.
- [PR #27](https://github.com/luca-heltai/immersx/pull/27) Added the initial coupled elasticity problem.

### Changed

- [PR #48](https://github.com/luca-heltai/immersx/pull/48) Updated the FVM headers.
- [PR #25](https://github.com/luca-heltai/immersx/pull/25) Updated the Docker environment and the 1D solver library download.

### Fixed

- [PR #49](https://github.com/luca-heltai/immersx/pull/49) Fixed compiler warnings.
- [PR #47](https://github.com/luca-heltai/immersx/pull/47) Removed the duplicated mass matrix from elasticity.
- [PR #46](https://github.com/luca-heltai/immersx/pull/46) Added missing VTK compile flags.
- [PR #43](https://github.com/luca-heltai/immersx/pull/43) Fixed the scaling of the Reduced Poisson problem.
- [PR #30](https://github.com/luca-heltai/immersx/pull/30) Addressed failing tests.

### Testing and CI

- [PR #33](https://github.com/luca-heltai/immersx/pull/33) Added an inclusion-orientation regression test for the solution and multiplier norms.
- [PR #28](https://github.com/luca-heltai/immersx/pull/28) Added the Lorthois datasets.

## [0.1] - 2024-11-07

### Added

- [PR #22](https://github.com/luca-heltai/immersx/pull/22) Added a geometric multigrid matrix-free preconditioner for the elasticity block.
- [PR #20](https://github.com/luca-heltai/immersx/pull/20) Added a matrix-free `CouplingOperator` for the action and transpose of the restriction operator.
- [PR #19](https://github.com/luca-heltai/immersx/pull/19) Added a matrix-free bulk evaluator.
- [PR #18](https://github.com/luca-heltai/immersx/pull/18) Enabled distributed parallel runs.
- [PR #13](https://github.com/luca-heltai/immersx/pull/13) Added local elasticity assembly.
- [PR #11](https://github.com/luca-heltai/immersx/pull/11) Added Dirichlet-only elasticity boundary conditions.
- [PR #5](https://github.com/luca-heltai/immersx/pull/5) Added Neumann and Robin boundary conditions.

### Changed

- [PR #23](https://github.com/luca-heltai/immersx/pull/23) Updated the license and repository metadata.
- [PR #8](https://github.com/luca-heltai/immersx/pull/8) Split declarations and definitions into separate files.
- [PR #6](https://github.com/luca-heltai/immersx/pull/6) Moved the `ReferenceInclusion` struct into its own header.

### Fixed

- [PR #17](https://github.com/luca-heltai/immersx/pull/17) Provided the locally relevant row index set when building distributed sparsity patterns.
- [PR #12](https://github.com/luca-heltai/immersx/pull/12) Fixed rotation handling.

### Testing and CI

- [PR #14](https://github.com/luca-heltai/immersx/pull/14) Added rotation tests.
