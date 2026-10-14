// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools_rhs.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/state.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <typename Type, typename = void>
  struct has_boundary_normal : std::false_type
  {};

  template <typename Type>
  struct has_boundary_normal<
    Type,
    std::void_t<decltype(std::declval<const Type &>().normal())>>
    : std::true_type
  {};

  template <typename Type, typename = void>
  struct has_boundary_conormal : std::false_type
  {};

  template <typename Type>
  struct has_boundary_conormal<
    Type,
    std::void_t<decltype(std::declval<const Type &>().conormal())>>
    : std::true_type
  {};

  template <typename Type, typename = void>
  struct has_boundary_tangent : std::false_type
  {};

  template <typename Type>
  struct has_boundary_tangent<
    Type,
    std::void_t<decltype(std::declval<const Type &>().tangent())>>
    : std::true_type
  {};

  static_assert(has_boundary_normal<BoundaryIntegrationSupport<2>>::value);
  static_assert(has_boundary_conormal<BoundaryIntegrationSupport<2, 3>>::value);
  static_assert(has_boundary_tangent<BoundaryIntegrationSupport<2>>::value);
  static_assert(!has_boundary_normal<BoundaryIntegrationSupport<2, 3>>::value);
  static_assert(!has_boundary_conormal<BoundaryIntegrationSupport<2>>::value);
  static_assert(!has_boundary_tangent<BoundaryIntegrationSupport<3>>::value);

  struct SquareLaw
  {
    struct Evaluation
    {
      double              value;
      std::vector<double> derivatives;
    };

    Evaluation
    evaluate(const std::vector<double> &values) const
    {
      AssertDimension(values.size(), 1);
      return {values[0] * values[0], {2. * values[0]}};
    }
  };

  class XBoundaryNormal : public Function<2>
  {
  public:
    double
    value(const Point<2> &point, const unsigned int = 0) const override
    {
      if (std::abs(point[0]) < 1.e-12)
        return -1.;
      if (std::abs(point[0] - 1.) < 1.e-12)
        return 1.;
      return 0.;
    }
  };

  struct ScalarSpace
  {
    ScalarSpace(Triangulation<2> &tria, const FiniteElement<2> &fe)
      : dof_handler(tria)
    {
      dof_handler.distribute_dofs(fe);
      constraints.close();
    }

    DoFHandler<2>             dof_handler;
    AffineConstraints<double> constraints;
  };

  template <typename FieldType>
  Vector<double>
  expected_scalar_pairing(const FieldType &field, const Vector<double> &source)
  {
    Vector<double>  result(field.dof_handler().n_dofs());
    const QGauss<2> quadrature(field.space().finite_element().degree + 1);
    FEValues<2>     fe_values(field.mapping(),
                          field.space().finite_element(),
                          quadrature,
                          update_values | update_JxW_values);
    std::vector<types::global_dof_index> indices(
      field.space().finite_element().n_dofs_per_cell());
    for (const auto &cell : field.dof_handler().active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(indices);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int i = 0; i < indices.size(); ++i)
            for (unsigned int j = 0; j < indices.size(); ++j)
              result[indices[i]] += fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) *
                                    source[indices[j]] * fe_values.JxW(q);
      }
    return result;
  }
} // namespace

TEST(WeakTerm, ScalarSameDoFHandler)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto                          term = weak_term(source, test(target));
  EXPECT_EQ(term(builder), target.field_id());

  Vector state(space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < state.size(); ++i)
    state[i] = 1. + i;
  Vector            residual(state.size());
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);
  model.evaluate_row(target.field_id(), context, residual);

  const auto expected = expected_scalar_pairing(source, state);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);

  const auto matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector action(state.size());
  matrix->view.vmult(action, state);
  action -= expected;
  EXPECT_LT(action.l2_norm(), 1.e-12);
}

TEST(WeakTerm, ExplicitCellSupportFiltersMaterialIds)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(1);
  for (const auto &cell : tria.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);

  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  const IntegrationSupport<2>         left_half(tria, 1);
  Model                               direct_model;
  SemidiscreteBuilder<Vector, Matrix> direct_builder(layout, direct_model);
  weak_term(value(source), test(target), left_half).add(direct_builder);

  Vector source_values(space.dof_handler.n_dofs());
  source_values = 1.;
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_values);
  const EvaluationContext<Vector> context(0., state_view);

  Vector direct_residual(target.dof_handler().n_dofs());
  direct_model.evaluate_row(target.field_id(), context, direct_residual);
  double integrated_measure = 0.;
  for (const auto value : direct_residual)
    integrated_measure += value;
  EXPECT_NEAR(integrated_measure, 0.5, 1.e-12);

  Model                               chained_model;
  SemidiscreteBuilder<Vector, Matrix> chained_builder(layout, chained_model);
  weak_term(value(source), test(target)).on(left_half).add(chained_builder);
  Vector chained_residual(target.dof_handler().n_dofs());
  chained_model.evaluate_row(target.field_id(), context, chained_residual);
  chained_residual -= direct_residual;
  EXPECT_LT(chained_residual.l2_norm(), 1.e-12);
}

TEST(WeakTerm, ExplicitBoundarySupportMatchesBoundaryRegion)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(1);

  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");

  using Vector = Vector<double>;
  using Matrix = SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  const BoundaryIntegrationSupport<2> support(tria, 0);
  const auto tangential_gradient = gradient(source) * support.tangent();
  const auto tangential_flux     = tangential_gradient * test(target) * support;
  (void)tangential_flux;
  Model                               explicit_model;
  SemidiscreteBuilder<Vector, Matrix> explicit_builder(layout, explicit_model);
  weak_term(value(source), test(target), support).add(explicit_builder);

  Model                               region_model;
  SemidiscreteBuilder<Vector, Matrix> region_builder(layout, region_model);
  weak_term(value(source), test(target)).on_boundary(0).add(region_builder);

  Model                               sugar_model;
  SemidiscreteBuilder<Vector, Matrix> sugar_builder(layout, sugar_model);
  (value(source) * test(target) * support).add(sugar_builder);

  Vector source_values(space.dof_handler.n_dofs());
  source_values = 1.;
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_values);
  const EvaluationContext<Vector> context(0., state_view);
  Vector explicit_residual(target.dof_handler().n_dofs());
  Vector region_residual(target.dof_handler().n_dofs());
  Vector sugar_residual(target.dof_handler().n_dofs());
  explicit_model.evaluate_row(target.field_id(), context, explicit_residual);
  region_model.evaluate_row(target.field_id(), context, region_residual);
  sugar_model.evaluate_row(target.field_id(), context, sugar_residual);

  region_residual -= explicit_residual;
  EXPECT_LT(region_residual.l2_norm(), 1.e-12);
  sugar_residual -= explicit_residual;
  EXPECT_LT(sugar_residual.l2_norm(), 1.e-12);
  region_residual = explicit_residual;
  Functions::ConstantFunction<2> forcing(1.);
  Vector native_boundary_rhs(target.dof_handler().n_dofs());
  VectorTools::create_boundary_right_hand_side(StaticMappingQ1<2>::mapping,
                                               target.dof_handler(),
                                               QGauss<1>(2),
                                               forcing,
                                               native_boundary_rhs,
                                               {0});
  region_residual -= native_boundary_rhs;
  EXPECT_LT(region_residual.l2_norm(), 1.e-12);
  EXPECT_NEAR(explicit_residual.l1_norm(), 4., 1.e-12);

  const auto normal_direction  = BoundaryIntegrationSupport<2>(tria).normal();
  const auto tangent_direction = BoundaryIntegrationSupport<2>(tria).tangent();
  FEFaceValues<2> face_values(StaticMappingQ1<2>::mapping,
                              fe,
                              QGauss<1>(2),
                              update_normal_vectors);
  bool            found_boundary_face = false;
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto face : cell->face_indices())
      if (cell->face(face)->at_boundary())
        {
          face_values.reinit(cell, face);
          const auto normal  = normal_direction.evaluate(face_values, 0);
          const auto tangent = tangent_direction.evaluate(face_values, 0);
          EXPECT_NEAR(normal.norm(), 1., 1.e-12);
          EXPECT_NEAR(tangent.norm(), 1., 1.e-12);
          EXPECT_NEAR(normal * tangent, 0., 1.e-12);
          found_boundary_face = true;
          break;
        }
  EXPECT_TRUE(found_boundary_face);
}

TEST(WeakTerm, BoundaryNormalFluxExpressionMatchesVectorTools)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(1);
  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");

  using Vector = Vector<double>;
  using Matrix = SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  const BoundaryIntegrationSupport<2> support(tria, 0);
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  (gradient(source) * support.normal() * test(target) * support).add(builder);

  std::vector<Point<2>> support_points(space.dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(StaticMappingQ1<2>::mapping,
                                       space.dof_handler,
                                       support_points);
  Vector source_values(space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < source_values.size(); ++i)
    source_values[i] = support_points[i][0];

  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_values);
  const EvaluationContext<Vector> context(0., state_view);
  Vector                          immersed_rhs(target.dof_handler().n_dofs());
  model.evaluate_row(target.field_id(), context, immersed_rhs);

  XBoundaryNormal forcing;
  Vector          native_rhs(target.dof_handler().n_dofs());
  VectorTools::create_boundary_right_hand_side(StaticMappingQ1<2>::mapping,
                                               target.dof_handler(),
                                               QGauss<1>(2),
                                               forcing,
                                               native_rhs,
                                               {0});
  immersed_rhs -= native_rhs;
  EXPECT_LT(immersed_rhs.l2_norm(), 1.e-12);
}

TEST(WeakTerm, EmbeddedBoundaryConormalExpressionCompiles)
{
  Triangulation<1, 2> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  FE_Q<1, 2>       fe(1);
  DoFHandler<1, 2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  StateLayout layout;
  const auto  space =
    fe_space(dof_handler, StaticMappingQ1<1, 2>::mapping, constraints);
  const auto                             source = space.field(layout, "source");
  const auto                             target = space.field(layout, "target");
  const BoundaryIntegrationSupport<1, 2> support(tria);
  const auto                             conormal_flux =
    gradient(source) * support.conormal() * test(target) * support;
  (void)conormal_flux;
}

TEST(WeakTerm, NonlinearSquareLawResidualAndJacobian)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto nonlinear = transform(value(source), SquareLaw{});
  weak_term(nonlinear, test(target)).add(builder);

  Vector state(source.dof_handler().n_dofs());
  for (unsigned int i = 0; i < state.size(); ++i)
    state[i] = 1. + 0.25 * i;
  Vector            target_state(target.dof_handler().n_dofs());
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  state_view.bind(target.field_id(), target_state);
  const EvaluationContext<Vector> context(0., state_view);

  Vector residual(target.dof_handler().n_dofs());
  model.evaluate_row(target.field_id(), context, residual);

  Vector                               expected(residual.size());
  const QGauss<2>                      quadrature(fe.degree + 1);
  FEValues<2>                          values(StaticMappingQ1<2>::mapping,
                     fe,
                     quadrature,
                     update_values | update_JxW_values);
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  for (const auto &cell : source.dof_handler().active_cell_iterators())
    {
      values.reinit(cell);
      cell->get_dof_indices(indices);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          double u = 0.;
          for (unsigned int j = 0; j < indices.size(); ++j)
            u += state[indices[j]] * values.shape_value(j, q);
          for (unsigned int i = 0; i < indices.size(); ++i)
            expected[indices[i]] +=
              u * u * values.shape_value(i, q) * values.JxW(q);
        }
    }
  const Vector residual_value = residual;
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);

  const auto jacobian =
    model.state_operator(target.field_id(), source.field_id(), context);
  Vector action(state.size());
  jacobian.vmult(action, state);
  Vector expected_action(action.size());
  for (const auto &cell : source.dof_handler().active_cell_iterators())
    {
      values.reinit(cell);
      cell->get_dof_indices(indices);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          double u  = 0.;
          double du = 0.;
          for (unsigned int j = 0; j < indices.size(); ++j)
            {
              u += state[indices[j]] * values.shape_value(j, q);
              du += state[indices[j]] * values.shape_value(j, q);
            }
          for (unsigned int i = 0; i < indices.size(); ++i)
            expected_action[indices[i]] +=
              2. * u * du * values.shape_value(i, q) * values.JxW(q);
        }
    }
  action -= expected_action;
  EXPECT_LT(action.l2_norm(), 1.e-12);

  const double epsilon   = 1.e-7;
  Vector       perturbed = state;
  perturbed.add(epsilon, state);
  StateView<Vector> perturbed_view(layout, 0.);
  perturbed_view.bind(source.field_id(), perturbed);
  perturbed_view.bind(target.field_id(), target_state);
  const EvaluationContext<Vector> perturbed_context(0., perturbed_view);
  Vector                          finite_difference(residual.size());
  model.evaluate_row(target.field_id(), perturbed_context, finite_difference);
  finite_difference -= residual_value;
  finite_difference *= 1. / epsilon;
  Vector jacobian_action(state.size());
  jacobian.vmult(jacobian_action, state);
  finite_difference -= jacobian_action;
  EXPECT_LT(finite_difference.l2_norm(), 1.e-6);
}

TEST(WeakTerm, ActiveAndFrozenObservablesUseTheirOwnFiniteElements)
{
  Triangulation<1, 3> tria;
  GridGenerator::subdivided_hyper_cube(tria, 2, 0., 1.);

  FE_Q<1, 3>       active_fe(1);
  FE_DGQ<1, 3>     frozen_fe(0);
  DoFHandler<1, 3> active_dh(tria), frozen_dh(tria);
  active_dh.distribute_dofs(active_fe);
  frozen_dh.distribute_dofs(frozen_fe);
  AffineConstraints<double> active_constraints, frozen_constraints;
  active_constraints.close();
  frozen_constraints.close();

  const auto active_space =
    fe_space(active_dh, StaticMappingQ1<1, 3>::mapping, active_constraints);
  const auto frozen_space =
    fe_space(frozen_dh, StaticMappingQ1<1, 3>::mapping, frozen_constraints);
  StateLayout layout;
  const auto  area     = active_space.field(layout, "area");
  const auto  target   = active_space.field(layout, "target");
  const auto  a0_field = frozen_space.field("a0");

  Vector<double> a0(frozen_dh.n_dofs());
  for (const auto &cell : frozen_dh.active_cell_iterators())
    {
      const auto                           id = cell->id();
      std::vector<types::global_dof_index> indices(frozen_fe.n_dofs_per_cell());
      cell->get_dof_indices(indices);
      a0[indices.front()] = id == frozen_dh.begin_active()->id() ? 1. : 4.;
    }
  const auto A0 = ImmersX::frozen(a0_field, a0);

  using Vector = Vector<double>;
  using Matrix = SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);

  const auto radius       = sqrt(value(area) / numbers::PI);
  const auto a0_radius    = sqrt(A0 / numbers::PI);
  const auto displacement = radius - a0_radius;
  const auto reversed_sum = a0_radius + radius;
  EXPECT_EQ(displacement.dependencies(), std::vector<FieldId>{area.field_id()});

  weak_term(displacement, test(target)).add(builder);

  Vector state(active_dh.n_dofs());
  state = 9.;
  Vector            target_state(active_dh.n_dofs());
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(area.field_id(), state);
  state_view.bind(target.field_id(), target_state);
  const EvaluationContext<Vector> context(0., state_view);

  Vector residual(active_dh.n_dofs());
  model.evaluate_row(target.field_id(), context, residual);

  Vector                               expected(residual.size());
  const QGauss<1>                      quadrature(2);
  FEValues<1, 3>                       values(StaticMappingQ1<1, 3>::mapping,
                        active_fe,
                        quadrature,
                        update_values | update_quadrature_points |
                          update_JxW_values);
  FEValues<1, 3>                       a0_values(StaticMappingQ1<1, 3>::mapping,
                           frozen_fe,
                           quadrature,
                           update_values);
  std::vector<types::global_dof_index> active_indices(
    active_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> frozen_indices(
    frozen_fe.n_dofs_per_cell());
  for (const auto &cell : active_dh.active_cell_iterators())
    {
      values.reinit(cell);
      const auto frozen_cell = cell->as_dof_handler_iterator(frozen_dh);
      a0_values.reinit(frozen_cell);
      cell->get_dof_indices(active_indices);
      frozen_cell->get_dof_indices(frozen_indices);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          double active_value = 0.;
          for (unsigned int j = 0; j < active_indices.size(); ++j)
            active_value += state[active_indices[j]] * values.shape_value(j, q);
          const double reference_area = a0[frozen_indices.front()];
          const double direct         = std::sqrt(active_value / numbers::PI) -
                                std::sqrt(reference_area / numbers::PI);
          const double direct_sum = std::sqrt(active_value / numbers::PI) +
                                    std::sqrt(reference_area / numbers::PI);
          const double expression_sum = reversed_sum.evaluate_point(
            values.quadrature_point(q),
            0.,
            [&](const auto &input, const auto &, const double) {
              if (input.is_frozen())
                return input.scale() * reference_area;
              return input.scale() * active_value;
            });
          EXPECT_NEAR(expression_sum, direct_sum, 1.e-14);
          for (unsigned int i = 0; i < active_indices.size(); ++i)
            expected[active_indices[i]] +=
              direct * values.shape_value(i, q) * values.JxW(q);
        }
    }
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);

  const auto jacobian =
    model.state_operator(target.field_id(), area.field_id(), context);
  Vector direction(state.size());
  for (unsigned int i = 0; i < direction.size(); ++i)
    direction[i] = 0.5 + 0.1 * i;
  Vector action(state.size());
  jacobian.vmult(action, direction);

  const double epsilon = 1.e-7;
  state.add(epsilon, direction);
  Vector perturbed(active_dh.n_dofs());
  model.evaluate_row(target.field_id(), context, perturbed);
  perturbed -= expected;
  perturbed *= 1. / epsilon;
  perturbed -= action;
  EXPECT_LT(perturbed.l2_norm(), 1.e-6);
}

TEST(WeakTerm, TypedVectorConvectionResidualJacobian)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FESystem<2> fe(FE_Q<2>(1), 2);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source", FEValuesExtractors::Vector(0));
  const auto target = V.field(layout, "target", FEValuesExtractors::Vector(0));
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto                          convection = gradient(source) * source;
  weak_term(convection, test(target)).add(builder);

  Vector state(source.dof_handler().n_dofs());
  Vector direction(state.size());
  for (unsigned int i = 0; i < state.size(); ++i)
    {
      state[i]     = 0.2 + 0.13 * i;
      direction[i] = -0.1 + 0.07 * i;
    }
  Vector            target_state(target.dof_handler().n_dofs());
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  state_view.bind(target.field_id(), target_state);
  const EvaluationContext<Vector> context(0., state_view);

  Vector residual(target.dof_handler().n_dofs());
  model.evaluate_row(target.field_id(), context, residual);

  const QGauss<2> quadrature(fe.degree + 1);
  FEValues<2>     values(StaticMappingQ1<2>::mapping,
                     fe,
                     quadrature,
                     update_values | update_gradients | update_JxW_values);
  const auto     &view = values[FEValuesExtractors::Vector(0)];
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  Vector                               expected(residual.size());
  for (const auto &cell : source.dof_handler().active_cell_iterators())
    {
      values.reinit(cell);
      cell->get_dof_indices(indices);
      for (const unsigned int q : values.quadrature_point_indices())
        {
          Tensor<1, 2> u;
          Tensor<2, 2> grad_u;
          for (unsigned int j = 0; j < indices.size(); ++j)
            {
              u += state[indices[j]] * view.value(j, q);
              grad_u += state[indices[j]] * view.gradient(j, q);
            }
          const auto convective = grad_u * u;
          for (unsigned int i = 0; i < indices.size(); ++i)
            expected[indices[i]] +=
              scalar_product(convective, view.value(i, q)) * values.JxW(q);
        }
    }
  const Vector residual_value = residual;
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);

  const auto jacobian =
    model.state_operator(target.field_id(), source.field_id(), context);
  Vector action(state.size());
  jacobian.vmult(action, direction);
  Vector expected_action(action.size());
  for (const auto &cell : source.dof_handler().active_cell_iterators())
    {
      values.reinit(cell);
      cell->get_dof_indices(indices);
      for (const unsigned int q : values.quadrature_point_indices())
        {
          Tensor<1, 2> u;
          Tensor<2, 2> grad_u;
          Tensor<1, 2> du;
          Tensor<2, 2> grad_du;
          for (unsigned int j = 0; j < indices.size(); ++j)
            {
              u += state[indices[j]] * view.value(j, q);
              grad_u += state[indices[j]] * view.gradient(j, q);
              du += direction[indices[j]] * view.value(j, q);
              grad_du += direction[indices[j]] * view.gradient(j, q);
            }
          const auto directional = grad_du * u + grad_u * du;
          for (unsigned int i = 0; i < indices.size(); ++i)
            expected_action[indices[i]] +=
              scalar_product(directional, view.value(i, q)) * values.JxW(q);
        }
    }
  action -= expected_action;
  EXPECT_LT(action.l2_norm(), 1.e-12);

  const double epsilon = 1.e-7;
  Vector       plus    = state;
  plus.add(epsilon, direction);
  StateView<Vector> plus_view(layout, 0.);
  plus_view.bind(source.field_id(), plus);
  plus_view.bind(target.field_id(), target_state);
  const EvaluationContext<Vector> plus_context(0., plus_view);
  Vector                          plus_residual(residual.size());
  model.evaluate_row(target.field_id(), plus_context, plus_residual);
  plus_residual -= residual_value;
  plus_residual *= 1. / epsilon;
  plus_residual -= expected_action;
  EXPECT_LT(plus_residual.l2_norm(), 1.e-6);
}

TEST(WeakTerm, ScalarDifferentDoFHandlersOnOneTriangulation)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FE_Q<2>     source_fe(1);
  FE_Q<2>     target_fe(2);
  ScalarSpace source_space(tria, source_fe);
  ScalarSpace target_space(tria, target_fe);
  StateLayout layout;
  const auto  source_view = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto  target_view = fe_space(target_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);
  const auto  source      = source_view.field(layout, "source");
  const auto  target      = target_view.field(layout, "target");
  using Vector            = Vector<double>;
  using Matrix            = SparseMatrix<double>;
  using Model             = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(value(source), test(target)).add(builder);

  Vector state(source_space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < state.size(); ++i)
    state[i] = 2. + i;
  Vector            residual(target_space.dof_handler.n_dofs());
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);
  model.evaluate_row(target.field_id(), context, residual);

  EXPECT_GT(residual.l2_norm(), 0.);
  const auto matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector action(residual.size());
  matrix->view.vmult(action, state);
  Vector jacobian_action = action;
  action -= residual;
  EXPECT_LT(action.l2_norm(), 1.e-12);

  const double epsilon         = 1.e-7;
  Vector       perturbed_state = state;
  perturbed_state.add(epsilon, state);
  StateView<Vector> perturbed_view(layout, 0.);
  perturbed_view.bind(source.field_id(), perturbed_state);
  const EvaluationContext<Vector> perturbed_context(0., perturbed_view);
  Vector                          perturbed_residual(residual.size());
  model.evaluate_row(target.field_id(), perturbed_context, perturbed_residual);
  perturbed_residual -= residual;
  perturbed_residual *= 1. / epsilon;
  perturbed_residual -= jacobian_action;
  EXPECT_LT(perturbed_residual.l2_norm(), 1.e-6);
}

TEST(WeakTerm, SameDoFHandlerGradientMatchesMatrixCreator)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(gradient(source), gradient(test(target))).add(builder);
  StateView<Vector>               state_view(layout, 0.);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      actual =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(actual.has_value());
  ASSERT_TRUE(actual->is_materializable());

  const QGauss<2>        quadrature(fe.degree + 1);
  DynamicSparsityPattern dynamic_sparsity(space.dof_handler.n_dofs(),
                                          space.dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(space.dof_handler,
                                  dynamic_sparsity,
                                  space.constraints,
                                  false);
  SparsityPattern sparsity;
  sparsity.copy_from(dynamic_sparsity);
  Matrix reference(sparsity);
  MatrixCreator::create_laplace_matrix(StaticMappingQ1<2>::mapping,
                                       space.dof_handler,
                                       quadrature,
                                       reference,
                                       static_cast<const Function<2> *>(
                                         nullptr),
                                       space.constraints);

  const auto actual_matrix = actual->matrix();
  ASSERT_EQ(actual_matrix->m(), reference.m());
  ASSERT_EQ(actual_matrix->n(), reference.n());
  for (unsigned int i = 0; i < reference.m(); ++i)
    for (unsigned int j = 0; j < reference.n(); ++j)
      EXPECT_NEAR(actual_matrix->el(i, j), reference.el(i, j), 1.e-12);
}

TEST(WeakTerm, PoissonLaplaceFormUsesTrialAndTestGradient)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto u = V.field(layout, "u");

  using Vector = Vector<double>;
  using Matrix = SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);

  // This is the weak form of the scalar Poisson operator.  In particular,
  // the test expression is derived from the same field as the trial
  // expression, so no solver-specific facade is required.
  weak_term(gradient(u), gradient(test(u))).add(builder);

  StateView<Vector>               state_view(layout, 0.);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      actual =
    model.state_matrix_operator(u.field_id(), u.field_id(), context);
  ASSERT_TRUE(actual.has_value());
  ASSERT_TRUE(actual->is_materializable());

  DynamicSparsityPattern dynamic_sparsity(space.dof_handler.n_dofs(),
                                          space.dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(space.dof_handler,
                                  dynamic_sparsity,
                                  space.constraints,
                                  false);
  SparsityPattern sparsity;
  sparsity.copy_from(dynamic_sparsity);
  Matrix reference(sparsity);
  MatrixCreator::create_laplace_matrix(StaticMappingQ1<2>::mapping,
                                       space.dof_handler,
                                       QGauss<2>(fe.degree + 1),
                                       reference,
                                       static_cast<const Function<2> *>(
                                         nullptr),
                                       space.constraints);

  const auto actual_matrix = actual->matrix();
  ASSERT_EQ(actual_matrix->m(), reference.m());
  ASSERT_EQ(actual_matrix->n(), reference.n());
  for (unsigned int i = 0; i < reference.m(); ++i)
    for (unsigned int j = 0; j < reference.n(); ++j)
      EXPECT_NEAR(actual_matrix->el(i, j), reference.el(i, j), 1.e-12);
}

TEST(WeakTerm, SameDoFHandlerVectorGradientUsesScalarProduct)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FESystem<2> fe(FE_Q<2>(1), 2);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source", FEValuesExtractors::Vector(0));
  const auto target = V.field(layout, "target", FEValuesExtractors::Vector(0));
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(gradient(source), gradient(test(target))).add(builder);
  StateView<Vector>               state_view(layout, 0.);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      actual =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(actual.has_value());
  ASSERT_TRUE(actual->is_materializable());

  const QGauss<2>        quadrature(fe.degree + 1);
  DynamicSparsityPattern dynamic_sparsity(space.dof_handler.n_dofs(),
                                          space.dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(space.dof_handler,
                                  dynamic_sparsity,
                                  space.constraints,
                                  false);
  SparsityPattern sparsity;
  sparsity.copy_from(dynamic_sparsity);
  Matrix      reference(sparsity);
  FEValues<2> values(StaticMappingQ1<2>::mapping,
                     fe,
                     quadrature,
                     update_gradients | update_JxW_values);
  const auto &view = values[FEValuesExtractors::Vector(0)];
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  FullMatrix<double> local(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  for (const auto &cell : space.dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      values.reinit(cell);
      cell->get_dof_indices(indices);
      local = 0.;
      for (const unsigned int q : values.quadrature_point_indices())
        for (unsigned int i = 0; i < indices.size(); ++i)
          for (unsigned int j = 0; j < indices.size(); ++j)
            local(i, j) +=
              dealii::scalar_product(view.gradient(j, q), view.gradient(i, q)) *
              values.JxW(q);
      space.constraints.distribute_local_to_global(local, indices, reference);
    }

  const auto actual_matrix = actual->matrix();
  for (unsigned int i = 0; i < reference.m(); ++i)
    for (unsigned int j = 0; j < reference.n(); ++j)
      EXPECT_NEAR(actual_matrix->el(i, j), reference.el(i, j), 1.e-12);
}

TEST(WeakTerm, SameDoFHandlerSymmetricGradientUsesBothExpressions)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FESystem<2> fe(FE_Q<2>(1), 2);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source", FEValuesExtractors::Vector(0));
  const auto target = V.field(layout, "target", FEValuesExtractors::Vector(0));
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(symmetric_gradient(source), symmetric_gradient(test(target)))
    .add(builder);
  StateView<Vector>               state_view(layout, 0.);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      actual =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(actual.has_value());
  ASSERT_TRUE(actual->is_materializable());

  const QGauss<2>        quadrature(fe.degree + 1);
  DynamicSparsityPattern dynamic_sparsity(space.dof_handler.n_dofs(),
                                          space.dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(space.dof_handler,
                                  dynamic_sparsity,
                                  space.constraints,
                                  false);
  SparsityPattern sparsity;
  sparsity.copy_from(dynamic_sparsity);
  Matrix      reference(sparsity);
  FEValues<2> values(StaticMappingQ1<2>::mapping,
                     fe,
                     quadrature,
                     update_gradients | update_JxW_values);
  const auto &view = values[FEValuesExtractors::Vector(0)];
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  FullMatrix<double> local(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  for (const auto &cell : space.dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      values.reinit(cell);
      cell->get_dof_indices(indices);
      local = 0.;
      for (const unsigned int q : values.quadrature_point_indices())
        for (unsigned int i = 0; i < indices.size(); ++i)
          for (unsigned int j = 0; j < indices.size(); ++j)
            local(i, j) +=
              dealii::scalar_product(view.symmetric_gradient(j, q),
                                     view.symmetric_gradient(i, q)) *
              values.JxW(q);
      space.constraints.distribute_local_to_global(local, indices, reference);
    }

  const auto actual_matrix = actual->matrix();
  for (unsigned int i = 0; i < reference.m(); ++i)
    for (unsigned int j = 0; j < reference.n(); ++j)
      EXPECT_NEAR(actual_matrix->el(i, j), reference.el(i, j), 1.e-12);
}

TEST(WeakTerm, TestSideDivergenceUsesTypedOperation)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FESystem<2> fe(FE_Q<2>(1), 3);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source =
    V.field(layout, "pressure", FEValuesExtractors::Scalar(0));
  const auto target =
    V.field(layout, "velocity", FEValuesExtractors::Vector(1));
  using Vector = Vector<double>;
  using Matrix = SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(value(source), divergence(test(target))).add(builder);
  StateView<Vector>               state_view(layout, 0.);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      actual =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(actual.has_value());
  ASSERT_TRUE(actual->is_materializable());

  const QGauss<2>        quadrature(fe.degree + 1);
  DynamicSparsityPattern dynamic_sparsity(space.dof_handler.n_dofs(),
                                          space.dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(space.dof_handler,
                                  dynamic_sparsity,
                                  space.constraints,
                                  false);
  SparsityPattern sparsity;
  sparsity.copy_from(dynamic_sparsity);
  Matrix      reference(sparsity);
  FEValues<2> values(StaticMappingQ1<2>::mapping,
                     fe,
                     quadrature,
                     update_values | update_gradients | update_JxW_values);
  const auto &source_view = values[FEValuesExtractors::Scalar(0)];
  const auto &target_view = values[FEValuesExtractors::Vector(1)];
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  FullMatrix<double> local(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  for (const auto &cell : space.dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      values.reinit(cell);
      cell->get_dof_indices(indices);
      local = 0.;
      for (const unsigned int q : values.quadrature_point_indices())
        for (unsigned int i = 0; i < indices.size(); ++i)
          for (unsigned int j = 0; j < indices.size(); ++j)
            local(i, j) += source_view.value(j, q) *
                           target_view.divergence(i, q) * values.JxW(q);
      space.constraints.distribute_local_to_global(local, indices, reference);
    }

  const auto actual_matrix = actual->matrix();
  for (unsigned int i = 0; i < reference.m(); ++i)
    for (unsigned int j = 0; j < reference.n(); ++j)
      EXPECT_NEAR(actual_matrix->el(i, j), reference.el(i, j), 1.e-12);
}

TEST(WeakTerm, ScaledObservableScalesResidualAndJacobian)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source");
  const auto target = V.field(layout, "target");
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(2.5 * value(source), 3.0 * test(target)).add(builder);

  Vector state(space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < state.size(); ++i)
    state[i] = 1. + i;
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);

  Vector residual(state.size());
  model.evaluate_row(target.field_id(), context, residual);
  auto expected = expected_scalar_pairing(source, state);
  expected *= 7.5;
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);

  const auto matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector action(state.size());
  matrix->view.vmult(action, state);
  action -= expected;
  EXPECT_LT(action.l2_norm(), 1.e-12);
}

TEST(WeakTerm, FrozenObservableContributesWithoutSourceDependency)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<2>     fe(1);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field("frozen-source");
  const auto target = V.field(layout, "target");
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Vector frozen_values(space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < frozen_values.size(); ++i)
    frozen_values[i] = 2. + i;
  const auto observable = frozen(source, frozen_values);
  EXPECT_TRUE(observable.dependencies().empty());
  EXPECT_TRUE(observable.is_frozen());
  EXPECT_FALSE(observable.source_field().is_valid());

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(observable, test(target)).add(builder);

  StateView<Vector>               state_view(layout, 0.);
  const EvaluationContext<Vector> context(0., state_view);
  Vector                          residual(space.dof_handler.n_dofs());
  model.evaluate_row(target.field_id(), context, residual);

  const auto expected = expected_scalar_pairing(source, frozen_values);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);
}

TEST(WeakTerm, VectorPairingHasConsistentTranspose)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FESystem<2> fe(FE_Q<2>(1), 2);
  ScalarSpace space(tria, fe);
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto source = V.field(layout, "source", FEValuesExtractors::Vector(0));
  const auto target = V.field(layout, "target", FEValuesExtractors::Vector(0));
  using Vector      = Vector<double>;
  using Matrix      = SparseMatrix<double>;
  using Model       = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(value(source), test(target)).add(builder);

  Vector state(space.dof_handler.n_dofs());
  Vector dual(space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < state.size(); ++i)
    {
      state[i] = 1. + i;
      dual[i]  = 3. - 0.25 * i;
    }
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector forward(state.size());
  Vector transpose(state.size());
  matrix->view.vmult(forward, state);
  matrix->view.Tvmult(transpose, dual);
  EXPECT_NEAR(forward * dual, state * transpose, 1.e-12);
}

TEST(WeakTerm, ScalarGradientPairsWithVectorTestField)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FE_Q<2>     source_fe(1);
  FESystem<2> target_fe(FE_Q<2>(1), 2);
  ScalarSpace source_space(tria, source_fe);
  ScalarSpace target_space(tria, target_fe);
  StateLayout layout;
  const auto  source_view = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto  target_view = fe_space(target_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    target_space.constraints);
  const auto  source      = source_view.field(layout, "source");
  const auto  target =
    target_view.field(layout, "target", FEValuesExtractors::Vector(0));
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(gradient(source), test(target)).add(builder);

  Vector state(source_space.dof_handler.n_dofs());
  for (unsigned int i = 0; i < state.size(); ++i)
    state[i] = 1. + i;
  Vector            residual(target_space.dof_handler.n_dofs());
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);
  model.evaluate_row(target.field_id(), context, residual);
  EXPECT_GT(residual.l2_norm(), 0.);

  const auto matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector action(residual.size());
  matrix->view.vmult(action, state);
  action -= residual;
  EXPECT_LT(action.l2_norm(), 1.e-12);
}

TEST(WeakTerm, MPI_DistributedSameDoFHandler)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);
  FE_Q<2>       fe(1);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();

  const auto V =
    fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);
  StateLayout layout;
  const auto  source = V.field(layout, "source");
  const auto  target = V.field(layout, "target");
  using Vector       = ImmersXLA::MPI::Vector;
  using Matrix       = ImmersXLA::MPI::SparseMatrix;
  using Model        = SemiDiscreteModel<Vector, Matrix>;

  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  weak_term(value(source), test(target)).add(builder);

  Vector state;
  state.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
  for (const auto index : state.locally_owned_elements())
    state[index] = 1. + index;
  state.compress(VectorOperation::insert);

  Vector residual;
  residual.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);
  model.evaluate_row(target.field_id(), context, residual);

  const auto matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector action;
  action.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
  matrix->view.vmult(action, state);
  action -= residual;
  EXPECT_LT(action.l2_norm(), 1.e-12);
}
