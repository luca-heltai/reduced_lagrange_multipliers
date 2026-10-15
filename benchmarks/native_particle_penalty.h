// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2026 by the ImmersX authors
// Independent deal.II oracle: this file must not use ImmersX assembly APIs.
#ifndef immersx_native_particle_penalty_h
#define immersx_native_particle_penalty_h

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/particles/particle_handler.h>

#include <immersx/compatibility/dealii_9_9/dof_handler_coupling.h>

namespace NativeParticlePenalty
{
  using namespace dealii;
#ifdef DEAL_II_WITH_TRILINOS
  using Matrix = LinearAlgebraTrilinos::MPI::SparseMatrix;
  using Vector = LinearAlgebraTrilinos::MPI::Vector;
#else
  using Matrix = LinearAlgebraPETSc::MPI::SparseMatrix;
  using Vector = LinearAlgebraPETSc::MPI::Vector;
#endif

  template <int support_dim, int spacedim, bool vector_valued>
  struct Fixture
  {
    parallel::distributed::Triangulation<spacedim> background{MPI_COMM_WORLD};
    // shared::Triangulation also supports lines in 2D/3D; each integration
    // cell still has exactly one owner and generates quadrature only there.
    parallel::shared::Triangulation<support_dim, spacedim> immersed{
      MPI_COMM_WORLD,
      Triangulation<support_dim, spacedim>::none,
      false,
      parallel::shared::Triangulation<support_dim, spacedim>::partition_zorder};
    FESystem<spacedim> fe{FE_Q<spacedim>(1), vector_valued ? spacedim : 1};
    FE_Q<support_dim, spacedim>             geometry_fe{1};
    DoFHandler<spacedim>                    dh{background};
    DoFHandler<support_dim, spacedim>       geometry_dh{immersed};
    AffineConstraints<double>               constraints;
    QGauss<support_dim>                     quadrature{3};
    IndexSet                                relevant;
    std::unique_ptr<DynamicSparsityPattern> sparsity;
    const double                            penalty               = 2.75;
    unsigned int                            active_particle_ranks = 0;
    double                                  measure               = 0.;

    Fixture(const unsigned int refinements)
    {
      GridGenerator::hyper_cube(background, -1., 1.);
      background.refine_global(refinements);
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            bool first_cell = true;
            for (unsigned int d = 0; d < spacedim; ++d)
              first_cell &= cell->center()[d] < -1. + 2. / (1u << refinements);
            if (first_cell)
              cell->set_refine_flag();
          }
      background.execute_coarsening_and_refinement();
      dh.distribute_dofs(fe);
      relevant = DoFTools::extract_locally_relevant_dofs(dh);
      constraints.reinit(dh.locally_owned_dofs(), relevant);
      DoFTools::make_hanging_node_constraints(dh, constraints);
      constraints.close();

      // Offset and tilted geometry crossing background ownership boundaries.
      std::vector<Point<spacedim>> vertices(
        GeometryInfo<support_dim>::vertices_per_cell);
      for (unsigned int v = 0; v < vertices.size(); ++v)
        {
          vertices[v][0] = -0.71;
          vertices[v][1] = -0.57;
          if constexpr (spacedim == 3)
            vertices[v][2] = -0.61;
          for (unsigned int d = 0; d < support_dim; ++d)
            if (v & (1u << d))
              for (unsigned int k = 0; k < spacedim; ++k)
                vertices[v][k] +=
                  k == d ? 1.17 : (k >= support_dim ? 0.91 / (d + 1) : 0.19);
        }
      CellData<support_dim> cell;
      for (unsigned int v = 0; v < vertices.size(); ++v)
        cell.vertices[v] = v;
      immersed.create_triangulation(vertices, {cell}, SubCellData());
      immersed.refine_global(1);
      geometry_dh.distribute_dofs(geometry_fe);
      sparsity = std::make_unique<DynamicSparsityPattern>(dh.n_dofs(),
                                                          dh.n_dofs(),
                                                          relevant);
      DoFTools::make_sparsity_pattern(dh, *sparsity, constraints, false);
      SparsityTools::distribute_sparsity_pattern(*sparsity,
                                                 dh.locally_owned_dofs(),
                                                 MPI_COMM_WORLD,
                                                 relevant);
    }

    void
    initialize(Matrix &matrix) const
    {
      matrix.reinit(dh.locally_owned_dofs(),
                    dh.locally_owned_dofs(),
                    *sparsity,
                    MPI_COMM_WORLD);
    }

    void
    explicit_particles(Matrix &matrix)
    {
      initialize(matrix);
      Particles::ParticleHandler<spacedim> particles(
        background, StaticMappingQ1<spacedim>::mapping, 1);
      FEValues<support_dim, spacedim> values(
        StaticMappingQ1<support_dim, spacedim>::mapping,
        geometry_fe,
        quadrature,
        update_quadrature_points | update_JxW_values);
      std::vector<Point<spacedim>>     points;
      std::vector<std::vector<double>> properties;
      double                           local_measure = 0.;
      for (const auto &cell : geometry_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                points.push_back(values.quadrature_point(q));
                properties.push_back({values.JxW(q)});
                local_measure += values.JxW(q);
              }
          }
      measure          = Utilities::MPI::sum(local_measure, MPI_COMM_WORLD);
      const auto boxes = Utilities::MPI::all_gather(
        MPI_COMM_WORLD,
        GridTools::compute_mesh_predicate_bounding_box(
          background, IteratorFilters::LocallyOwnedCell()));
      // Construct locations and weights manually; let deal.II perform the
      // distributed point search and transfer to background-cell owners.
      particles.insert_global_particles(points, boxes, properties);
      AssertThrow(
        particles.n_global_particles() ==
          immersed.n_global_active_cells() * quadrature.size(),
        ExcMessage(
          "Manual particle insertion lost or duplicated quadrature points."));
      active_particle_ranks =
        Utilities::MPI::sum(static_cast<unsigned int>(
                              particles.n_locally_owned_particles() > 0),
                            MPI_COMM_WORLD);
      AssertThrow(
        active_particle_ranks >=
          std::min(2u, Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        ExcMessage(
          "The fixture must exercise multiple background particle owners."));
      std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
      FullMatrix<double> local(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
      auto               particle = particles.begin();
      while (particle != particles.end())
        {
          local           = 0;
          const auto cell = particle->get_surrounding_cell();
          typename DoFHandler<spacedim>::cell_iterator dh_cell(*cell, &dh);
          dh_cell->get_dof_indices(indices);
          const auto in_cell = particles.particles_in_cell(cell);
          for (const auto &p : in_cell)
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                if (fe.system_to_component_index(i).first ==
                    fe.system_to_component_index(j).first)
                  local(i, j) += penalty *
                                 fe.shape_value(i, p.get_reference_location()) *
                                 fe.shape_value(j, p.get_reference_location()) *
                                 p.get_properties()[0];
          constraints.distribute_local_to_global(local, indices, matrix);
          particle = in_cell.end();
        }
      matrix.compress(VectorOperation::add);
    }

    void
    coupling_matrix(Matrix &matrix) const
    {
      initialize(matrix);
      NonMatching::DoFHandlerCoupling<support_dim, spacedim> coupling(
        dh, geometry_dh);
      coupling.create_nitsche_restriction_matrix(quadrature,
                                                 penalty,
                                                 matrix,
                                                 constraints);
      matrix.compress(VectorOperation::add);
      AssertThrow(
        coupling.get_quadrature_particle_handler().n_global_particles() ==
          immersed.n_global_active_cells() * quadrature.size(),
        ExcMessage("DoFHandlerCoupling lost or duplicated quadrature points."));
    }
  };
} // namespace NativeParticlePenalty
#endif
