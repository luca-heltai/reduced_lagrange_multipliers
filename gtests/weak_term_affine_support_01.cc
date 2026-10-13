// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <typename VectorType,
            typename MatrixType,
            typename FieldType,
            typename Oracle>
  void
  check_affine_model(SemiDiscreteModel<VectorType, MatrixType> &model,
                     const StateLayout                         &layout,
                     const FieldType                           &source,
                     const FieldType                           &target,
                     const Oracle                              &oracle)
  {
    VectorType state, direction, residual, expected, shifted, action;
    detail::initialize_weak_vector(state,
                                   source.locally_owned_dofs(),
                                   source.space().mpi_communicator());
    detail::initialize_weak_vector(direction,
                                   source.locally_owned_dofs(),
                                   source.space().mpi_communicator());
    for (const auto index : source.locally_owned_dofs())
      if (!source.constraints().is_constrained(index))
        {
          state[index]     = 0.3 + 0.01 * index;
          direction[index] = 0.7 - 0.003 * index;
        }
    if constexpr (detail::has_vector_compress<VectorType>::value)
      {
        state.compress(VectorOperation::insert);
        direction.compress(VectorOperation::insert);
      }
    for (auto *vector : {&residual, &expected, &shifted, &action})
      detail::initialize_weak_vector(*vector,
                                     target.locally_owned_dofs(),
                                     target.space().mpi_communicator());
    StateView<VectorType> view(layout, 0.);
    view.bind(source.field_id(), state);
    EvaluationContext<VectorType> context(0., view);
    model.evaluate_row(target.field_id(), context, residual);
    oracle(expected, 1.);
    EXPECT_GT(expected.l2_norm(), 1.e-5);
    expected -= residual;
    EXPECT_LT(expected.l2_norm(), 1.e-12);

    const auto jacobian = model.state_matrix_operator(target.field_id(),
                                                      source.field_id(),
                                                      context);
    ASSERT_TRUE(jacobian.has_value());
    jacobian->view.vmult(action, direction);
    const double epsilon = 1.e-6;
    state.add(epsilon, direction);
    model.evaluate_row(target.field_id(), context, shifted);
    shifted -= residual;
    shifted *= 1. / epsilon;
    shifted -= action;
    EXPECT_LT(shifted.l2_norm(), 1.e-9);

    state    = 0.;
    residual = 0.;
    expected = 0.;
    model.evaluate_row(target.field_id(), context, residual);
    oracle(expected, 0.);
    expected -= residual;
    EXPECT_LT(expected.l2_norm(), 1.e-12);
  }

  // Reconstruct the physical FE coefficients independently of weak-term
  // storage.
  template <typename FieldType>
  Vector<double>
  physical_state(const FieldType &source, const double free_scale)
  {
    Vector<double> state(source.locally_owned_dofs().size());
    for (unsigned int i = 0; i < state.size(); ++i)
      state[i] = free_scale * (0.3 + 0.01 * i);
    source.constraints().distribute(state);
    return state;
  }

  template <typename VectorType, typename FieldType>
  void
  cell_oracle(VectorType                  &expected,
              const FieldType             &source,
              const FieldType             &target,
              const IntegrationSupport<2> &support,
              const double                 free_scale)
  {
    const auto  state = physical_state(source, free_scale);
    FEValues<2> trial(source.mapping(),
                      source.space().finite_element(),
                      QGauss<2>(4),
                      update_values);
    FEValues<2> test_values(target.mapping(),
                            target.space().finite_element(),
                            QGauss<2>(4),
                            update_values | update_JxW_values);
    for (const auto &cell : source.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned() && support.includes_cell(cell))
        {
          const auto target_cell =
            cell->as_dof_handler_iterator(target.dof_handler());
          trial.reinit(cell);
          test_values.reinit(target_cell);
          std::vector<double> values(trial.n_quadrature_points);
          trial.get_function_values(state, values);
          Vector<double> local(target_cell->get_fe().n_dofs_per_cell());
          std::vector<types::global_dof_index> indices(local.size());
          target_cell->get_dof_indices(indices);
          Vector<double> diagonal(local.size());
          for (unsigned int q = 0; q < values.size(); ++q)
            for (unsigned int i = 0; i < local.size(); ++i)
              {
                const double phi = test_values.shape_value(i, q);
                local[i] += phi * values[q] * test_values.JxW(q);
                diagonal[i] += phi * phi * test_values.JxW(q);
              }
          target.constraints().distribute_local_to_global(local,
                                                          indices,
                                                          expected);
          // deal.II square assembly also enforces constrained diagonal rows.
          // The execution state has zero constrained entries in this test.
          if (&source.constraints() == &target.constraints() &&
              &source.dof_handler() == &target.dof_handler())
            for (unsigned int i = 0; i < indices.size(); ++i)
              if (source.constraints().is_constrained(indices[i]))
                expected[indices[i]] -=
                  diagonal[i] *
                  source.constraints().get_inhomogeneity(indices[i]);
        }
    detail::compress_weak_vector(expected);
  }
} // namespace

TEST(WeakTerm, AffineCellSupportRectangularAndDynamic)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       source_fe(2), target_fe(1);
  DoFHandler<2> source_dofs(mesh), target_dofs(mesh);
  source_dofs.distribute_dofs(source_fe);
  target_dofs.distribute_dofs(target_fe);
  AffineConstraints<double> source_constraints, target_constraints;
  source_constraints.add_line(0);
  source_constraints.set_inhomogeneity(0, 2.);
  source_constraints.add_line(1);
  source_constraints.add_entry(1, 2, 0.5);
  source_constraints.set_inhomogeneity(1, 0.4);
  source_constraints.close();
  target_constraints.add_line(3);
  target_constraints.add_entry(3, 4, 0.25);
  target_constraints.set_inhomogeneity(
    3, 7.); // Test offsets must not enter the residual.
  target_constraints.close();
  const auto source_space =
    fe_space(source_dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto target_space =
    fe_space(target_dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout                 layout;
  const auto                  source = source_space.field(layout, "source");
  const auto                  target = target_space.field(layout, "target");
  const IntegrationSupport<2> support(mesh, 1);
  using V = Vector<double>;
  using M = SparseMatrix<double>;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(source), test(target), support).add(builder);
  for (const double offset : {2., -1., 0.})
    {
      source_constraints.set_inhomogeneity(0, offset);
      check_affine_model(model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           cell_oracle(
                             expected, source, target, support, free_scale);
                         });
    }
}

TEST(WeakTerm, BOTH_AffineCellSupportHangingAndAsymmetric)
{
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(2);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 0.3 &&
        cell->center()[1] < 0.3)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  for (const auto &cell : mesh.active_cell_iterators())
    if (!cell->is_artificial())
      cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
  AffineConstraints<double> constraints(dofs.locally_owned_dofs(), relevant);
  DoFTools::make_hanging_node_constraints(dofs, constraints);
  EXPECT_GT(Utilities::MPI::sum(constraints.n_constraints(), MPI_COMM_WORLD),
            0u);
  const auto points =
    DoFTools::map_dofs_to_support_points(StaticMappingQ1<2>::mapping, dofs);
  types::global_dof_index affine_node = numbers::invalid_dof_index;
  // This fine-cell interior node contributes to cells with hanging test rows.
  const Point<2> affine_point(0.125, 0.125);
  for (const auto &[index, point] : points)
    if (relevant.is_element(index) && point.distance(affine_point) < 1.e-12)
      {
        affine_node = index;
        constraints.add_line(index);
        constraints.set_inhomogeneity(index, 0.);
      }
  constraints.close();
  EXPECT_FALSE(constraints.has_inhomogeneities());
  const auto space = fe_space(dofs, StaticMappingQ1<2>::mapping, constraints);
  EXPECT_EQ(space.locally_relevant_dofs(), relevant);
  StateLayout layout;
  const auto  source = space.field(layout, "source");
  const auto  target = space.field(layout, "target");
  using V            = ImmersXLA::MPI::Vector;
  using M            = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   explicit_model, inferred_model;
  SemidiscreteBuilder<V, M> explicit_builder(layout, explicit_model),
    inferred_builder(layout, inferred_model);
  const IntegrationSupport<2> support(mesh, 1), whole_mesh(mesh);
  weak_term(value(source), test(target), support).add(explicit_builder);
  weak_term(value(source), test(target)).add(inferred_builder);
  for (const double offset : {2., -3., 0.})
    {
      if (affine_node != numbers::invalid_dof_index)
        constraints.set_inhomogeneity(affine_node, offset);
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 2)
        EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                        constraints.has_inhomogeneities()),
                                      MPI_COMM_WORLD),
                  offset == 0. ? 0u : 1u);
      check_affine_model(explicit_model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           cell_oracle(
                             expected, source, target, support, free_scale);
                         });
      check_affine_model(inferred_model,
                         layout,
                         source,
                         target,
                         [&](V &expected, const double free_scale) {
                           cell_oracle(
                             expected, source, target, whole_mesh, free_scale);
                         });
    }
}

TEST(WeakTerm, AffineFrozenAndConstraintCellSupports)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  AffineConstraints<double> source_constraints, target_constraints;
  source_constraints.add_line(0);
  source_constraints.set_inhomogeneity(0, 2.);
  source_constraints.close();
  target_constraints.close();
  const auto source_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, source_constraints);
  const auto target_space =
    fe_space(dofs, StaticMappingQ1<2>::mapping, target_constraints);
  StateLayout layout;
  const auto  source = source_space.field(layout, "source");
  const auto  target = target_space.field(layout, "multiplier");
  using V            = Vector<double>;
  using M            = SparseMatrix<double>;
  V coefficients(dofs.n_dofs());
  for (unsigned int i = 0; i < coefficients.size(); ++i)
    if (!source_constraints.is_constrained(i))
      coefficients[i] = 0.3 + 0.01 * i;
  const auto observable = frozen(source, coefficients);
  EXPECT_TRUE(observable.dependencies().empty());
  SemiDiscreteModel<V, M>   frozen_model, constraint_model;
  SemidiscreteBuilder<V, M> frozen_builder(layout, frozen_model),
    constraint_builder(layout, constraint_model);
  const IntegrationSupport<2> cells(mesh, 1);
  weak_term(observable, test(target), cells).add(frozen_builder);
  const double sign = -1.25;
  weak_term(value(source), test(target), cells)
    .add_constraint_terms(constraint_builder, target.field_id(), sign, 0);
  StateView<V>         view(layout, 0.);
  EvaluationContext<V> context(0., view);
  for (const double offset : {2., -1., 0.})
    {
      source_constraints.set_inhomogeneity(0, offset);
      V residual(dofs.n_dofs()), expected(dofs.n_dofs());
      frozen_model.evaluate_row(target.field_id(), context, residual);
      cell_oracle(expected, source, target, cells, 1.);
      EXPECT_GT(expected.l2_norm(), 1.e-5);
      expected -= residual;
      EXPECT_LT(expected.l2_norm(), 1.e-12);
      check_affine_model(constraint_model,
                         layout,
                         source,
                         target,
                         [&](V &result, const double free_scale) {
                           cell_oracle(
                             result, source, target, cells, free_scale);
                           result *= sign;
                         });
    }
}
