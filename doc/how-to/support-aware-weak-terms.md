# Choose where a weak term is integrated

A weak term combines a finite-element expression, a test function, and a
geometric integration support. The support chooses the cells, faces, or
physical quadrature points used by the integral. It does not need to own a
finite-element space.

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
