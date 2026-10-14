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

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparse_matrix.h>

#include <gtest/gtest.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

TEST(WeakTermNonmatching, BOTH_RejectsUnsupportedAffineSourceOffsets)
{
  parallel::distributed::Triangulation<2> source_mesh(MPI_COMM_WORLD),
    target_mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(source_mesh);
  GridGenerator::hyper_cube(target_mesh);
  source_mesh.refine_global(1);
  target_mesh.refine_global(2);
  FE_Q<2>       fe(1);
  DoFHandler<2> source_dofs(source_mesh), target_dofs(target_mesh);
  source_dofs.distribute_dofs(fe);
  target_dofs.distribute_dofs(fe);
  const auto source_relevant =
    DoFTools::extract_locally_relevant_dofs(source_dofs);
  AffineConstraints<double> source_constraints(source_dofs.locally_owned_dofs(),
                                               source_relevant),
    target_constraints;
  if (source_relevant.is_element(0))
    source_constraints.add_line(0);
  source_constraints.close();
  target_constraints.close();
  const auto source_space =
    fe_space(source_dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto target_space =
    fe_space(target_dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout layout;
  const auto  source = source_space.field(layout, "source");
  const auto  target = target_space.field(layout, "target");
  using V            = ImmersXLA::MPI::Vector;
  using M            = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(source), test(target)).add(builder);
  V state, residual;
  detail::initialize_weak_vector(state,
                                 source.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(residual,
                                 target.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  state = 1.;
  state.compress(VectorOperation::insert);
  StateView<V> view(layout, 0.);
  view.bind(source.field_id(), state);
  EvaluationContext<V> context(0., view);
  EXPECT_NO_THROW(model.evaluate_row(target.field_id(), context, residual));

  // Only the owning rank changes the offset. Rejection must be collective.
  if (source.locally_owned_dofs().is_element(0))
    source_constraints.set_inhomogeneity(0, 2.);
  EXPECT_THROW(model.evaluate_row(target.field_id(), context, residual),
               dealii::ExceptionBase);
  SemiDiscreteModel<V, M>   rejected_model;
  SemidiscreteBuilder<V, M> rejected_builder(layout, rejected_model);
  EXPECT_THROW(weak_term(value(source), test(target)).add(rejected_builder),
               dealii::ExceptionBase);
}

TEST(WeakTerm, RejectsUnsupportedAffineMultiplierReaction)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> source_constraints, multiplier_constraints;
  source_constraints.close();
  multiplier_constraints.add_line(0);
  multiplier_constraints.set_inhomogeneity(0, 2.);
  multiplier_constraints.close();
  const auto source_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto multiplier_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, multiplier_constraints);
  StateLayout layout;
  const auto  source     = source_space.field(layout, "source");
  const auto  multiplier = multiplier_space.field(layout, "multiplier");
  SemiDiscreteModel<Vector<double>, SparseMatrix<double>>   model;
  SemidiscreteBuilder<Vector<double>, SparseMatrix<double>> builder(layout,
                                                                    model);
  EXPECT_THROW(weak_term(value(source), test(multiplier))
                 .add_constraint_terms(builder, multiplier.field_id(), 1., 0),
               dealii::ExceptionBase);
}
