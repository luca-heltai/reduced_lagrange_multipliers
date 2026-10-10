// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/core/constraint.h>
#  include <immersx/core/observable_lift.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/core/weak_term.h>
#  include <immersx/io/imported_finite_element_fields.h>
#  include <immersx/io/output_handler.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>

#  include <filesystem>
#  include <iostream>
#  include <stdexcept>
#  include <string>

namespace
{
  using namespace dealii;
  using namespace ImmersX;

  std::string
  metric_flow_parameter(const std::string &name)
  {
    auto &prm = ParameterAcceptor::prm;
    prm.enter_subsection("MetricFlowSystem<1, 3>");
    const auto result = prm.get(name);
    prm.leave_subsection();
    return result;
  }

  void
  run_coupled_problem(const std::string &parameter_file)
  {
    using SolidProblem = ElastodynamicsSolver<3>;
    using FlowProblem  = MetricFlowX::BloodFlowSystem<1, 3>;
    using FieldVector  = MetricFlowX::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = IDAAdapter<FieldVector, GlobalVector>;
    using SolidField   = Field<3, 3, FEValuesExtractors::Vector>;
    using WallLift     = TensorProductLift<1, 2, 3, 3>;

    ParameterAcceptor::clear();
    FlowProblem                 flow_problem(MPI_COMM_WORLD);
    TimeIntervalParameters      flow_time;
    FixedStepParameters         fixed_step;
    IDAParameters               ida_parameters;
    ElastodynamicsParameters<3> solid_parameters("/Elastodynamics/",
                                                 &flow_time,
                                                 &fixed_step,
                                                 &ida_parameters);
    WallLift                    wall_lift("/MetricFlowX vessel wall lift/");

    initialize_parameters(parameter_file);
    flow_problem.initialize_params(parameter_file);

    flow_problem.setup();
    const auto imported = std::make_shared<ImportedFiniteElementFields<1, 3>>(
      metric_flow_parameter("Vtk file path for mesh input"),
      flow_problem.triangulation(),
      MPI_COMM_WORLD);
    const auto a0 = imported->field("a0");
    const auto A0 = frozen(a0.field(), a0.coefficients());

    SolidProblem solid_problem(solid_parameters);
    solid_problem.make_grid();
    solid_problem.setup_fe();
    solid_problem.setup_system();
    solid_problem.assemble_operators();
    solid_problem.set_initial_conditions();

    Adapter    adapter(flow_time, ida_parameters, MPI_COMM_WORLD);
    const auto solid_fields = adapter.add(solid_problem, "elastodynamics");
    const auto flow_fields =
      adapter.add(metric_flow_x(flow_problem), "blood-flow");

    const auto solid_field       = solid_fields.fields().displacement;
    const auto area              = flow_fields.fields().area;
    const auto radius            = sqrt(A0 / numbers::PI);
    const auto dR                = sqrt(value(area) / numbers::PI) - radius;
    const auto wall_displacement = lift(dR, wall_lift, radius);
    const auto lambda_field =
      area.with_id(FieldId()).compacted("lambda",
                                        flow_fields.fields().area_components);
    const auto lambda_wall = lift(value(lambda_field), wall_lift, radius);
    const auto kinematic_constraint =
      make_constraint(weak_term(value(solid_field), test(lambda_wall)) -
                      weak_term(wall_displacement, test(lambda_field)));
    const auto coupling_fields =
      adapter.add(kinematic_constraint, "vessel-wall");

    adapter.add(weak_term(-1. * value(coupling_fields.fields().multiplier),
                          test(flow_fields.fields().velocity)),
                "flow-pressure-feedback");

    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    initialize_elastodynamics_adapter_state(
      adapter, solid_fields, solid_problem, state, state_dot);
    auto &flow_state = adapter.field(state, flow_fields.fields().state);
    flow_problem.initialize_state(flow_state, flow_time.initial_time);
    flow_problem.initialize_state_derivative(
      adapter.field(state_dot, flow_fields.fields().state),
      flow_time.initial_time);
    adapter.field(state, coupling_fields.fields().multiplier)     = 0.;
    adapter.field(state_dot, coupling_fields.fields().multiplier) = 0.;

    const auto flow_output_directory =
      metric_flow_parameter("Output directory");
    OutputHandlerParameters multiplier_output_parameters(
      "/MetricFlowX multiplier output/");
    multiplier_output_parameters.output_directory =
      flow_output_directory + "/interaction";
    OutputHandler<1, 3, FieldVector> multiplier_output(
      area.space(), multiplier_output_parameters, "pressure_jump");
    const auto output = [&solid_problem,
                         &solid_parameters,
                         &flow_problem,
                         &adapter,
                         &multiplier_output,
                         coupling_fields,
                         flow_fields,
                         flow_output_directory](const GlobalVector &solution,
                                                const unsigned int  step,
                                                const double        time) {
      std::filesystem::create_directories(flow_output_directory);
      std::filesystem::create_directories(solid_parameters.output_directory);
      const auto &flow_state =
        adapter.field(solution, flow_fields.fields().state);
      auto pressure = flow_problem.make_state();
      flow_problem.compute_pressure(flow_state, pressure, time);
      flow_problem.output_results(flow_state, pressure, step);
      solid_problem.output_results();
      multiplier_output.write_field(
        coupling_fields.fields().multiplier,
        adapter.field(solution, coupling_fields.fields().multiplier),
        time);
    };

    adapter.set_output_step([&solid_problem, &adapter, output, solid_fields](
                              const double        time,
                              const GlobalVector &solution,
                              const GlobalVector &solution_dot,
                              const unsigned int  step) {
      solid_problem.accept_state(
        adapter.field(solution, solid_fields.fields().displacement),
        adapter.field(solution, solid_fields.fields().velocity),
        time,
        step);
      output(solution, step, time);
      (void)solution_dot;
    });

    output(state, 0, flow_time.initial_time);
    const auto accepted_steps = adapter.solve(state, state_dot);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "MetricFlowX elastodynamics accepted IDA steps: "
                << accepted_steps << std::endl;
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);
      if (argc != 2)
        throw std::invalid_argument(
          "usage: app_metric_flow_x_elastodynamics PARAMETER_FILE");
      run_coupled_problem(argv[1]);
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception on processing: " << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
    }
  return 0;
}

#else

int
main()
{
  return 0;
}

#endif // IMMERSX_WITH_METRIC_FLOW_X
