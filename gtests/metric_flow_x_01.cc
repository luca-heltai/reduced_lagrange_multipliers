// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/imported_finite_element_fields.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/metric_flow_x.h>
#include <immersx/physics/metric_flow_x_pressure.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)

namespace
{
  using Problem      = MetricFlowX::BloodFlowSystem<1, 3>;
  using FieldVector  = MetricFlowX::VectorType;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;

  void
  initialize_problem(Problem &problem)
  {
    const auto parameter_file =
      ImmersX::TestPaths::parameter_path("gtests/parameters/metric_flow_x.prm");
    problem.initialize_params(parameter_file);
    problem.setup();
  }

  template <typename Fields>
  void
  initialize_adapter_state(Problem      &problem,
                           Adapter      &adapter,
                           const Fields &fields,
                           GlobalVector &state,
                           GlobalVector &state_dot)
  {
    problem.initialize_state(adapter.field(state, fields.fields().state), 0.);
    problem.initialize_state_derivative(adapter.field(state_dot,
                                                      fields.fields().state),
                                        0.);
  }

  void
  add_synthetic_residual(Adapter &adapter, const ImmersX::FieldId state)
  {
    adapter.add(
      [state](auto &builder) {
        auto term = builder.term(state, "synthetic");
        term.residual([state](const auto &context) {
          const auto *reference = &context.state(state);
          dealii::PackagedOperation<FieldVector> result;
          result.reinit_vector = [reference](FieldVector &vector, const bool) {
            vector.reinit(*reference);
          };
          result.apply     = [](FieldVector &vector) { vector = 2.5; };
          result.apply_add = [](FieldVector &vector) {
            FieldVector contribution;
            contribution.reinit(vector);
            contribution = 2.5;
            vector += contribution;
          };
          return result;
        });
        return state;
      },
      "synthetic");
  }
} // namespace

TEST(MetricFlowX, FeatureMacroIsEnabled)
{
  SUCCEED();
}

TEST(MetricFlowX, BOTH_RegistersOneMixedStateField)
{
  dealii::ParameterAcceptor::clear();
  Problem                         problem(MPI_COMM_WORLD);
  ImmersX::TimeIntervalParameters time_parameters(
    "/MetricFlowSystem<1, 3>/Time interval/");
  ImmersX::IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA/");
  initialize_problem(problem);

  Adapter    adapter(time_parameters, ida_parameters, MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");

  EXPECT_TRUE(fields.fields().state.is_valid());
  EXPECT_TRUE(adapter.has_local_preconditioner(fields.fields().state));
  EXPECT_EQ(adapter.differential_components(), problem.differential_dofs());
  EXPECT_EQ(fields.fields().area.name(), "area");
  EXPECT_EQ(fields.fields().velocity.name(), "velocity");
  EXPECT_EQ(&fields.fields().area.dof_handler(), &problem.dof_handler());
  EXPECT_EQ(&fields.fields().velocity.dof_handler(), &problem.dof_handler());
  EXPECT_EQ(fields.fields().area.component_mask(),
            problem.finite_element().component_mask(
              dealii::FEValuesExtractors::Scalar(0)));
  EXPECT_EQ(fields.fields().velocity.component_mask(),
            problem.finite_element().component_mask(
              dealii::FEValuesExtractors::Scalar(1)));
  auto state = adapter.make_state();
  EXPECT_EQ(state.n_blocks(), 1u);
}

TEST(MetricFlowX, BOTH_ResidualMatchesNativeAndPreservesAdditivity)
{
  dealii::ParameterAcceptor::clear();
  Problem                         problem(MPI_COMM_WORLD);
  ImmersX::TimeIntervalParameters time_parameters(
    "/MetricFlowSystem<1, 3>/Time interval/");
  ImmersX::IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA/");
  initialize_problem(problem);

  Adapter    adapter(time_parameters, ida_parameters, MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");
  add_synthetic_residual(adapter, fields.fields().state);
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  initialize_adapter_state(problem, adapter, fields, state, state_dot);

  auto native_residual = problem.make_state();
  problem.assemble_residual(0.,
                            adapter.field(state, fields.fields().state),
                            adapter.field(state_dot, fields.fields().state),
                            native_residual);

  GlobalVector residual;
  adapter.solver().reinit_vector(residual);
  adapter.solver().residual(0., state, state_dot, residual);
  auto difference = adapter.field(residual, fields.fields().state);
  difference -= native_residual;
  for (const auto index : problem.locally_owned_dofs())
    EXPECT_NEAR(difference[index], 2.5, 1.e-12);
}

TEST(MetricFlowX, BOTH_JacobianActionsMatchNativeMatrices)
{
  dealii::ParameterAcceptor::clear();
  Problem                         problem(MPI_COMM_WORLD);
  ImmersX::TimeIntervalParameters time_parameters(
    "/MetricFlowSystem<1, 3>/Time interval/");
  ImmersX::IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA/");
  initialize_problem(problem);

  Adapter    adapter(time_parameters, ida_parameters, MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  initialize_adapter_state(problem, adapter, fields, state, state_dot);

  auto direction      = adapter.make_state();
  direction           = 0.125;
  auto expected_state = problem.make_state();
  auto expected_dot   = problem.make_state();
  problem.assemble_state_jacobian(0.,
                                  adapter.field(state, fields.fields().state),
                                  adapter.field(state_dot,
                                                fields.fields().state));
  problem.state_jacobian_matrix().vmult(expected_state,
                                        adapter.field(direction,
                                                      fields.fields().state));
  problem.assemble_derivative_jacobian(
    0.,
    adapter.field(state, fields.fields().state),
    adapter.field(state_dot, fields.fields().state));
  problem.derivative_jacobian_matrix().vmult(
    expected_dot, adapter.field(direction, fields.fields().state));

  auto action = adapter.make_state();
  adapter.solver().setup_jacobian(0., state, state_dot, 0.);
  adapter.current_jacobian().vmult(action, direction);
  auto difference = adapter.field(action, fields.fields().state);
  difference -= expected_state;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);

  adapter.solver().setup_jacobian(0., state, state_dot, 1.);
  adapter.current_jacobian().vmult(action, direction);
  difference = adapter.field(action, fields.fields().state);
  difference -= expected_state;
  difference -= expected_dot;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);
}

TEST(MetricFlowX, BOTH_IDAVerticalSmoke)
{
  dealii::ParameterAcceptor::clear();
  Problem                         problem(MPI_COMM_WORLD);
  ImmersX::TimeIntervalParameters time_parameters(
    "/MetricFlowSystem<1, 3>/Time interval/");
  ImmersX::IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA/");
  initialize_problem(problem);

  Adapter    adapter(time_parameters, ida_parameters, MPI_COMM_WORLD);
  const auto fields =
    adapter.add(ImmersX::metric_flow_x(problem), "blood-flow");
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  initialize_adapter_state(problem, adapter, fields, state, state_dot);
  adapter.set_compute_consistent_initial_conditions(
    [&problem, &adapter, fields](const double  time,
                                 GlobalVector &state,
                                 GlobalVector &state_dot) {
      problem.initialize_state_derivative(adapter.field(state_dot,
                                                        fields.fields().state),
                                          time);
      (void)state;
    });

  const auto initial_state = state;
  EXPECT_GT(adapter.solve(state, state_dot), 0u);
  state -= initial_state;
  EXPECT_GT(state.l2_norm(), 1.e-12);
}

TEST(MetricFlowX, BOTH_NativeStateInitializesPositiveArea)
{
  dealii::ParameterAcceptor::clear();
  Problem                         problem(MPI_COMM_WORLD);
  ImmersX::TimeIntervalParameters time_parameters(
    "/MetricFlowSystem<1, 3>/Time interval/");
  ImmersX::IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA/");
  initialize_problem(problem);

  auto state = problem.make_state();
  problem.compute_initial_solution(state, 0.);

  double minimum_area = std::numeric_limits<double>::max();
  for (const auto index : problem.component_dofs(Problem::Component::area))
    if (state.locally_owned_elements().is_element(index))
      minimum_area = std::min(minimum_area, static_cast<double>(state[index]));
  EXPECT_GT(dealii::Utilities::MPI::min(minimum_area, MPI_COMM_WORLD), 0.);
}

TEST(MetricFlowX, BOTH_WallTractionActsOnMomentumAndPreservesContinuity)
{
  using namespace dealii;
  using namespace ImmersX;
  ParameterAcceptor::clear();
  Problem                            flow(MPI_COMM_WORLD);
  TimeIntervalParameters             time;
  FixedStepParameters                step;
  IDAParameters                      ida;
  ElastodynamicsParameters<3>        parameters("/Elastodynamics/",
                                         &time,
                                         &step,
                                         &ida);
  TensorProductLift<1, 2, 3, 3>      lift("/MetricFlowX vessel wall lift/");
  FiniteElementSpaceParameters<1, 3> multiplier_parameters(
    "/MetricFlowX multiplier space/", "FE_DGQ(2)");
  const auto input = TestPaths::parameter_path(
    "gtests/parameters/metric_flow_x_elastodynamics.prm");
  initialize_parameters(input);
  flow.initialize_params(input);
  flow.setup();
  multiplier_parameters.finite_element = "FE_DGQ(2)";
  ElastodynamicsSolver<3> solid(parameters);
  solid.make_grid();
  solid.setup_fe();
  solid.setup_system();
  solid.assemble_operators();
  solid.set_initial_conditions();
  FiniteElementSpace<1, 3> multiplier_space(flow.triangulation(),
                                            multiplier_parameters);
  const auto               multiplier_view = multiplier_space.view();
  const auto               multiplier      = multiplier_view.field("lambda");
  ImportedFiniteElementFields<1, 3> imported(
    TestPaths::data_filename("metric_flow_x/single_vessel_centered.vtk"),
    flow.triangulation(),
    MPI_COMM_WORLD);
  const auto a0 = imported.field("a0");
  const auto A0 = frozen(a0.field(), a0.coefficients());
  Adapter    adapter(time, ida, MPI_COMM_WORLD);
  adapter.set_schur_primal_coupling();
  const auto solid_fields = adapter.add(solid, "solid");
  const auto flow_fields  = adapter.add(metric_flow_x(flow), "flow");
  const auto wall         = adapter.add(
    [&](auto &builder) {
      const auto id =
        builder.algebraic_field(multiplier.name(),
                                multiplier.locally_owned_dofs(),
                                multiplier.locally_relevant_dofs());
      const auto lambda      = multiplier.with_id(id);
      const auto radius      = sqrt(A0 / numbers::PI);
      const auto lambda_wall = ImmersX::lift(value(lambda), lift, radius);
      const auto negative_wall_displacement =
        ImmersX::lift(radius -
                        sqrt(value(flow_fields.fields().area) / numbers::PI),
                      lift,
                      radius);
      weak_term(value(solid_fields.fields().displacement), test(lambda_wall))
        .add(builder);
      weak_term(negative_wall_displacement, test(lambda)).add(builder);
      weak_term(lambda_wall, test(solid_fields.fields().velocity)).add(builder);
      const auto pressure      = -1. * value(lambda);
      const auto flow_id       = flow_fields.fields().state;
      auto       pressure_term = builder.term(flow_id, "native-pressure");
      pressure_term.residual([&, pressure, flow_id, id](const auto &context) {
        auto contribution = std::make_shared<FieldVector>(flow.make_state());
        *contribution     = 0.;
        flow.add_external_pressure_residual(
          context.time(),
          context.state(flow_id),
          metric_flow_x_pressure_provider(pressure, context.state(id)),
          *contribution);
        PackagedOperation<FieldVector> result;
        result.reinit_vector = [&flow](auto &dst, const bool) {
          flow.reinit_state(dst);
        };
        result.apply     = [contribution](auto &dst) { dst = *contribution; };
        result.apply_add = [contribution](auto &dst) { dst += *contribution; };
        return result;
      });
      typename std::decay_t<decltype(builder)>::Model::MatrixOperatorFactory
        pressure_jacobian = [&, pressure, flow_id](const auto &context) {
          return metric_flow_x_pressure_operator(flow,
                                                 pressure,
                                                 context.state(flow_id),
                                                 context.time());
        };
      pressure_term.state(id, std::move(pressure_jacobian));
      builder.saddle_point(id,
                                   {solid_fields.fields().displacement.id(),
                            solid_fields.fields().velocity.id(),
                            flow_fields.fields().state},
                           detail::make_multiplier_metric(builder, lambda));
      return ConstraintFields<std::decay_t<decltype(multiplier)>>{lambda};
    },
    "wall");
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  flow.initialize_state(adapter.field(state, flow_fields.fields().state), 0.);
  flow.initialize_state_derivative(adapter.field(state_dot,
                                                 flow_fields.fields().state),
                                   0.);
  auto baseline = adapter.make_state();
  auto forced   = adapter.make_state();
  adapter.solver().residual(0., state, state_dot, baseline);
  adapter.field(state, wall.fields().multiplier) = 0.5;
  adapter.solver().residual(0., state, state_dot, forced);
  forced -= baseline;
  EXPECT_NEAR(
    adapter.field(forced, solid_fields.fields().displacement).l2_norm(),
    0.,
    1.e-14);
  EXPECT_GT(adapter.field(forced, solid_fields.fields().velocity).l2_norm(),
            1.e-10);
  EXPECT_NEAR(adapter.field(forced, wall.fields().multiplier).l2_norm(),
              0.,
              1.e-14);
  const auto &flow_increment =
    adapter.field(forced, flow_fields.fields().state);
  for (const auto index : flow.component_dofs(Problem::Component::area))
    if (flow.locally_owned_dofs().is_element(index))
      EXPECT_NEAR(flow_increment[index], 0., 1.e-14);

  // Check the complete state/derivative linearization at a nonzero multiplier
  // and an expanded area, so the nonlinear radius derivative is exercised.
  auto  direction      = adapter.make_state();
  auto &flow_direction = adapter.field(direction, flow_fields.fields().state);
  auto &flow_state     = adapter.field(state, flow_fields.fields().state);
  for (const auto index : flow.component_dofs(Problem::Component::area))
    if (flow.locally_owned_dofs().is_element(index))
      {
        flow_state[index] *= 1.1;
        flow_direction[index] = 0.1 * flow_state[index];
      }
  adapter.field(direction, wall.fields().multiplier) = 0.2;
  constexpr double alpha                             = 10.;
  adapter.solver().setup_jacobian(0., state, state_dot, alpha);
  auto action = adapter.make_state();
  adapter.current_jacobian().vmult(action, direction);
  constexpr double epsilon   = 1.e-5;
  auto             plus      = state;
  auto             minus     = state;
  auto             dot_plus  = state_dot;
  auto             dot_minus = state_dot;
  plus.add(epsilon, direction);
  minus.add(-epsilon, direction);
  dot_plus.add(alpha * epsilon, direction);
  dot_minus.add(-alpha * epsilon, direction);
  adapter.solver().residual(0., plus, dot_plus, forced);
  adapter.solver().residual(0., minus, dot_minus, baseline);
  forced -= baseline;
  forced *= 0.5 / epsilon;
  forced -= action;
  EXPECT_LT(forced.l2_norm(), 1.e-6 * action.l2_norm() + 1.e-12);

  auto small_rhs = action;
  small_rhs *= 1.e-14;
  auto correction = adapter.make_state();
  adapter.solver().solve_with_jacobian(small_rhs, correction, 1.e-10);
  adapter.current_jacobian().vmult(forced, correction);
  forced -= small_rhs;
  EXPECT_LT(forced.l2_norm(), 1.e-6 * small_rhs.l2_norm());
}

TEST(MetricFlowX, BOTH_IDAFailureDoesNotPublishAnAcceptedStep)
{
  dealii::ParameterAcceptor::clear();
  Problem                         problem(MPI_COMM_WORLD);
  ImmersX::TimeIntervalParameters time_parameters(
    "/MetricFlowSystem<1, 3>/Time interval/");
  ImmersX::IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA/");
  initialize_problem(problem);
  using IDA = ImmersX::Compatibility::SUNDIALS::IDA<FieldVector>;
  IDA::AdditionalData data;
  data.final_time                    = 1.e-4;
  data.ic_type                       = IDA::AdditionalData::none;
  data.reset_type                    = IDA::AdditionalData::none;
  data.maximum_non_linear_iterations = 2;
  data.maximum_convergence_failures  = 1;
  IDA solver(data, MPI_COMM_WORLD);
  solver.reinit_vector = [&problem](FieldVector &vector) {
    problem.reinit_state(vector);
  };
  solver.residual = [](double, const auto &, const auto &, auto &residual) {
    residual = 1.;
  };
  solver.setup_jacobian = [](double, const auto &, const auto &, double) {};
  solver.solve_with_jacobian = [](const auto &rhs, auto &solution, double) {
    solution = rhs;
  };
  unsigned int outputs = 0;
  solver.output_step =
    [&outputs](double, const auto &, const auto &, unsigned int) {
      // Bound the old Release failure path, which otherwise keeps publishing
      // the failed state and retrying indefinitely.
      if (++outputs > 1)
        throw std::runtime_error("A failed step was published");
    };
  auto state     = problem.make_state();
  auto state_dot = problem.make_state();
  EXPECT_THROW(solver.solve_dae(state, state_dot),
               ImmersX::Compatibility::SUNDIALS::ExcIDAError);
  EXPECT_EQ(outputs, 1u);
}

TEST(MetricFlowX, BOTH_PressureProviderPreservesDGTracesAndGhostValues)
{
  using namespace dealii;
  using namespace ImmersX;
  ParameterAcceptor::clear();
  Problem                problem(MPI_COMM_WORLD);
  TimeIntervalParameters time("/MetricFlowSystem<1, 3>/Time interval/");
  IDAParameters          ida("/MetricFlowSystem<1, 3>/IDA/");
  FiniteElementSpaceParameters<1, 3> parameters("/Pressure space/",
                                                "FE_DGQ(1)");
  problem.initialize_params(
    TestPaths::parameter_path("gtests/parameters/metric_flow_x_pressure.prm"));
  problem.setup();
  FiniteElementSpace<1, 3> space(problem.triangulation(), parameters);
  const auto               view  = space.view();
  const auto               field = view.field("pressure_jump");
  FieldVector              coefficients;
  coefficients.reinit(field.locally_owned_dofs(), MPI_COMM_WORLD);
  const auto                          &fe = space.finite_element();
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  for (const auto &cell : view.dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell->get_dof_indices(indices);
        for (unsigned int j = 0; j < indices.size(); ++j)
          {
            const auto x = view.mapping().transform_unit_to_real_cell(
              cell, fe.get_unit_support_points()[j]);
            coefficients[indices[j]] = 1. + 2. * x[0] - cell->center()[0];
          }
      }
  coefficients.compress(VectorOperation::insert);
  const auto provider =
    metric_flow_x_pressure_provider(-1. * value(field), coefficients);
  unsigned int local_ghosts = 0;
  for (const auto &cell : view.dof_handler().active_cell_iterators())
    if (!cell->is_artificial())
      {
        local_ghosts += cell->is_ghost();
        for (const double coordinate : {0., 0.5, 1.})
          {
            const auto point =
              view.mapping().transform_unit_to_real_cell(cell,
                                                         Point<1>(coordinate));
            EXPECT_NEAR(provider({0., point, cell->material_id(), cell->id()}),
                        -1. - 2. * point[0] + cell->center()[0],
                        1.e-12);
          }
      }
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    EXPECT_GT(Utilities::MPI::sum(local_ghosts, MPI_COMM_WORLD), 0u);

  auto state = problem.make_state();
  auto dot   = problem.make_state();
  problem.initialize_state(state, 0.);
  problem.initialize_state_derivative(dot, 0.);
  auto baseline     = problem.make_state();
  auto pressured    = problem.make_state();
  auto contribution = problem.make_state();
  problem.assemble_residual(0., state, dot, baseline);
  problem.set_external_pressure_provider(provider);
  problem.assemble_residual(0., state, dot, pressured);
  problem.clear_external_pressure_provider();
  pressured -= baseline;
  contribution = 0.;
  problem.add_external_pressure_residual(0., state, provider, contribution);
  auto error = pressured;
  error -= contribution;
  EXPECT_LE(error.l2_norm(), 1.e-10 * std::max(1., contribution.l2_norm()));
  EXPECT_GT(contribution.l2_norm(), 1.e-12);
  for (const auto index : problem.component_dofs(Problem::Component::area))
    if (contribution.locally_owned_elements().is_element(index))
      EXPECT_NEAR(contribution[index], 0., 1.e-14);

  const auto op =
    metric_flow_x_pressure_operator(problem, -1. * value(field), state, 0.);
  op.view.vmult(pressured, coefficients);
  pressured -= contribution;
  EXPECT_LE(pressured.l2_norm(), 1.e-13 * contribution.l2_norm());
  auto dual = problem.make_state();
  for (const auto i : dual.locally_owned_elements())
    dual[i] = 1. + 0.1 * i;
  auto transpose = coefficients;
  op.view.Tvmult(transpose, dual);
  EXPECT_NEAR(transpose * coefficients,
              dual * contribution,
              1.e-11 * std::max(1., std::abs(dual * contribution)));
}

#else

TEST(MetricFlowX, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
