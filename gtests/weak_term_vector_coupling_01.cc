// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparse_matrix.h>

#include <gtest/gtest.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int spacedim>
  void
  check_vector_coupling()
  {
    parallel::distributed::Triangulation<spacedim> background(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(background, -1., 1.);
    Triangulation<1, spacedim> embedded;
    GridGenerator::hyper_cube(embedded, -0.6, 0.6);
    FESystem<spacedim>      background_fe(FE_Q<spacedim>(1), spacedim + 2);
    FESystem<1, spacedim>   embedded_fe(FE_Q<1, spacedim>(1), spacedim + 2);
    DoFHandler<spacedim>    background_dh(background);
    DoFHandler<1, spacedim> embedded_dh(embedded);
    background_dh.distribute_dofs(background_fe);
    embedded_dh.distribute_dofs(embedded_fe);
    AffineConstraints<double> constraints;
    constraints.close();
    MappingQ1<spacedim>    background_mapping;
    MappingQ1<1, spacedim> embedded_mapping;
    const auto             background_space =
      fe_space(background_dh, background_mapping, constraints);
    const auto embedded_space =
      fe_space(embedded_dh, embedded_mapping, constraints);
    const auto b = value(background_space.field(FieldId(0),
                                                "background",
                                                FEValuesExtractors::Vector(1)));
    const auto e = value(embedded_space.field(FieldId(0),
                                              "embedded",
                                              FEValuesExtractors::Vector(1)));
    using Matrix = SparseMatrix<double>;
    using Vector = dealii::Vector<double>;
    const auto reverse =
      detail::WeakAssembly<decltype(b),
                           decltype(e)>::template assemble<Vector, Matrix>(b,
                                                                           e);
    const auto forward =
      detail::WeakAssembly<decltype(e),
                           decltype(b)>::template assemble<Vector, Matrix>(e,
                                                                           b);

    // Direct FEValuesViews quadrature is independent of component masks and
    // checks each matrix entry, including transverse and unselected components.
    const QGauss<1>       quadrature(2);
    FEValues<1, spacedim> embedded_values(embedded_mapping,
                                          embedded_fe,
                                          quadrature,
                                          update_values |
                                            update_quadrature_points |
                                            update_JxW_values);
    const auto            embedded_cell = embedded_dh.begin_active();
    embedded_values.reinit(embedded_cell);
    std::vector<Point<spacedim>> unit_points;
    const auto                   background_cell = background_dh.begin_active();
    for (const auto &point : embedded_values.get_quadrature_points())
      unit_points.push_back(
        background_mapping.transform_real_to_unit_cell(background_cell, point));
    FEValues<spacedim> background_values(background_mapping,
                                         background_fe,
                                         Quadrature<spacedim>(unit_points),
                                         update_values);
    background_values.reinit(background_cell);
    std::vector<types::global_dof_index> bi(background_fe.n_dofs_per_cell()),
      ei(embedded_fe.n_dofs_per_cell());
    background_cell->get_dof_indices(bi);
    embedded_cell->get_dof_indices(ei);
    const FEValuesExtractors::Vector vector(1);
    for (unsigned int i = 0; i < ei.size(); ++i)
      for (unsigned int j = 0; j < bi.size(); ++j)
        {
          double expected = 0.;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            expected += embedded_values[vector].value(i, q) *
                        background_values[vector].value(j, q) *
                        embedded_values.JxW(q);
          EXPECT_NEAR(reverse.matrix->el(ei[i], bi[j]), expected, 1.e-13);
          EXPECT_NEAR(forward.matrix->el(bi[j], ei[i]), expected, 1.e-13);
        }
  }
} // namespace

TEST(WeakTermNonmatching, NativeVectorCouplingPreservesAmbientComponents)
{
  check_vector_coupling<2>();
  check_vector_coupling<3>();
}
