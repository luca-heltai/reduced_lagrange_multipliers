# Choose where a weak term is integrated

A weak term combines a finite-element expression, a test function, and a
geometric integration support. The support chooses the cells, faces, or
physical quadrature points used by the integral. It does not need to own a
finite-element space.

## Configure a field space

Register space parameters before reading the application parameter file. The
owning space distributes DoFs on a Domain's distributed triangulation; retain
both the space and its view while Fields or execution adapters use them:

```cpp
DomainParameters<1, 2> domain_parameters("/Line/Domain/");
FiniteElementSpaceParameters<1, 2> multiplier_parameters(
  "/Line/Multiplier space/", "FE_DGQ<1>(0)");
initialize_parameters(parameter_file);
Domain<1, 2> line(domain_parameters);
line.make_grid();
FiniteElementSpace<1, 2> multiplier_space(line.triangulation(),
                                         multiplier_parameters);
auto multiplier_view = multiplier_space.view();
auto lambda = multiplier_view.field("lambda");
```

The `Finite element` entry accepts names understood by
`dealii::FETools::get_fe_by_name()`. Coupled Poisson exposes this entry under
`Coupled Poisson/Multiplier space`, defaulting to `FE_DGQ<1>(0)`. Its generated
regression input `gtests/parameters/coupled_poisson_multiplier_q1.prm` selects
`FE_DGQ<1>(1)` with nonzero loads.

Static elasticity exposes `Elastic static/Displacement space/Finite element`.
The default `auto` entry retains `FE degree` and builds `spacedim` copies of `FE_Q`.
For example, `FESystem<2>[FE_Q<2>(2)^2]` selects quadratic displacement in 2D.
The element must be primitive with `spacedim` components; assembly quadrature
uses the selected element's actual degree. Problems still own their native
matrices, boundary constraints, material data, and accepted solution.

For a new Problem, own its Domain and FiniteElementSpace, update the space's
constraints for the physical boundary data, and expose the native DoFHandler,
mapping, constraints, and relevant indices. Contributors can use
`finite_element_space_view(problem)` without repeating those accessors. Keep
native residual assembly when it carries affine boundary corrections; changing
ownership does not require replacing it with a weak term.

## Cell and boundary supports

Use a domain support to select all cells or one material id. Use a boundary
support to select all boundary faces or one boundary id:

```cpp
auto dOmega = domain.support(1);
auto mass = value(u) * test(v) * dOmega;
auto laplace = weak_term(gradient(u), gradient(test(v)))
                 .on(domain.support());

auto wall = domain.boundary_support(0);
auto boundary_mass = value(u) * test(v) * wall;
auto normal_flux = gradient(u) * wall.normal() * test(v) * wall;
```

The multiplication form and `weak_term(...).on(support)` form select the same
support. The matching-mesh path uses deal.II `FEValues` or `FEFaceValues`.
Matching linear terms retain affine source offsets and refresh changed
inhomogeneities; their meshes, mappings, and constraint entries must stay fixed.

## Integrate bulk fields on an independent mesh

An immersed domain can supply integration geometry for trial and test fields
that both live on a background mesh:

```cpp
auto dGamma = immersed_domain.support();
auto trace_mass = value(u_background) * test(v_background) * dGamma;
trace_mass.add(builder);
```

The immersed mesh needs no `DoFHandler` or `Field`. Its quadrature points and
physical weights are located in the background mesh, and each field is
evaluated on its own background `DoFHandler`. The current path supports
codimension-one supports and fields on the same background triangulation and
mapping. It uses `QGauss(p + 2)` on each support cell without splitting that
cell at background-cell boundaries, so refine the support mesh where it crosses
many background cells.

The same physical-point assembly accepts composed observables, including
nonlinear expressions with frozen finite-element inputs:

```cpp
// response is a composed observable of active and frozen fields.
weak_term(response, test(v_background)).on(dGamma).add(builder);
```

The residual and Jacobian use the current active state and the frozen input
coefficients. Frozen FE inputs must share the source mesh and mapping.
Independent and lifted linear pairings currently require homogeneous source
constraints and reject offsets changed after assembly.
Rebuild a mutable support after changing its geometric inputs.
The geometric shape derivative is not included in the weak-term Jacobian.

## Use a lifted support

`Domain::lift()` turns a representative mesh and a `TensorProductLift` into
physical quadrature support. It creates no finite-element space on the lifted
geometry:

```cpp
auto dV = centerline.lift(lift);
auto lifted_mass = value(u_background) * test(v_background) * dV;
lifted_mass.add(builder);
```

For lifted and independent immersed supports, trial and test fields must use
the same full-dimensional background triangulation and mapping, and all
support points must lie in its searchable domain. Geometry is fixed during
residual and Jacobian evaluation. Call `dV.rebuild()` after changing the mesh,
lift descriptor, or thickness evaluator. See [Integrate over a lifted
geometry](lifted-integration) for ownership, normal, and benchmark details.
