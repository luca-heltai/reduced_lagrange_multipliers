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

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/core/constraint.h>
#  include <immersx/core/domain.h>
#  include <immersx/core/observable_lift.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/core/weak_term.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>
#  include <immersx/physics/metric_flow_x_pressure.h>

#  include <filesystem>
#  include <fstream>
#  include <iostream>
#  include <stdexcept>
#  include <string>
#  include <type_traits>
#  include <vector>

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
    FiniteElementSpaceParameters<1, 3> lambda_parameters(
      "/MetricFlowX multiplier space/", "auto");

    ParameterAcceptor coupled_solver("/MetricFlowX coupled solver/");
    std::string       linear_solver = "iterative";
    coupled_solver.add_parameter(
      "Linear solver",
      linear_solver,
      "Coupled IDA linear solver. Direct solvers materialize each Jacobian. UMFPACK requires one rank; MUMPS supports MPI.",
      ParameterAcceptor::prm,
      Patterns::Selection("iterative|umfpack|mumps"));

    initialize_parameters(parameter_file);
    flow_problem.initialize_params(parameter_file);

    AssertThrow(linear_solver != "umfpack" ||
                  Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage(
                  "The coupled UMFPACK solver requires one MPI rank."));
#  ifndef DEAL_II_WITH_UMFPACK
    AssertThrow(
      linear_solver != "umfpack",
      ExcMessage(
        "The coupled UMFPACK solver requires deal.II configured with UMFPACK."));
#  endif
#  ifndef DEAL_II_WITH_MUMPS
    AssertThrow(
      linear_solver != "mumps",
      ExcMessage(
        "The coupled MUMPS solver requires deal.II configured with MUMPS."));
#  endif
    DomainParameters<1, 3> flow_domain_parameters("/MetricFlowX domain/");
    flow_domain_parameters.name_of_grid =
      metric_flow_parameter("Vtk file path for mesh input");
    flow_domain_parameters.read_vtk_fields = true;
    Domain<1, 3> flow_domain(flow_domain_parameters,
                             flow_problem.triangulation());
    flow_problem.setup(
      [&](const auto &coarse_mesh, const unsigned int refinement) {
        flow_domain_parameters.initial_refinement = refinement;
        flow_domain.prepare_grid(coarse_mesh);
        flow_domain.finalize_grid();
      });
    if (lambda_parameters.finite_element == "auto")
      lambda_parameters.finite_element =
        "FE_DGQ(" +
        std::to_string(flow_problem.finite_element().base_element(0).degree) +
        ")";
    FiniteElementSpace<1, 3> lambda_space(flow_problem.triangulation(),
                                          lambda_parameters);
    const auto              *area_fe = dynamic_cast<const FE_DGQ<1, 3> *>(
      &flow_problem.finite_element().base_element(0));
    const auto *lambda_fe =
      dynamic_cast<const FE_DGQ<1, 3> *>(&lambda_space.finite_element());
    AssertThrow(area_fe != nullptr && lambda_fe != nullptr &&
                  area_fe->degree == lambda_fe->degree &&
                  area_fe->n_dofs_per_cell() == lambda_fe->n_dofs_per_cell(),
                ExcMessage("The multiplier FE must have the same local "
                           "basis as MetricFlowX area."));
    const auto  lambda_view = lambda_space.view();
    const auto  lambda      = lambda_view.field("lambda");
    const auto  a0          = flow_domain.field("a0");
    FieldVector reference_area;
    reference_area.reinit(flow_domain.vtk_fields().locally_owned_dofs(),
                          flow_domain.vtk_fields().locally_relevant_dofs(),
                          MPI_COMM_WORLD);
    reference_area = a0.coefficients();
    reference_area.update_ghost_values();
    const auto A0 = frozen(a0.field(), reference_area);

    SolidProblem solid_problem(solid_parameters);
    solid_problem.make_grid();
    solid_problem.setup_fe();
    solid_problem.setup_system();
    solid_problem.assemble_operators();
    solid_problem.set_initial_conditions();

    Adapter adapter(flow_time, ida_parameters, MPI_COMM_WORLD);
    adapter.set_schur_primal_coupling();
    if (linear_solver == "iterative")
      adapter.set_linear_solver(Adapter::LinearSolver::iterative);
    else if (linear_solver == "umfpack")
      adapter.set_linear_solver(Adapter::LinearSolver::umfpack);
    else if (linear_solver == "mumps")
      adapter.set_linear_solver(Adapter::LinearSolver::mumps);
    const auto solid_fields = adapter.add(solid_problem, "elastodynamics");
    const auto flow_fields =
      adapter.add(metric_flow_x(flow_problem), "blood-flow");

    const auto coupling_fields = adapter.add(
      [&](auto &builder) {
        const auto id          = builder.algebraic_field(lambda.name(),
                                                lambda.locally_owned_dofs(),
                                                lambda.locally_relevant_dofs());
        const auto multiplier  = lambda.with_id(id);
        const auto radius      = sqrt(A0 / numbers::PI);
        const auto lambda_wall = lift(value(multiplier), wall_lift, radius);
        const auto negative_wall_displacement =
          lift(radius - sqrt(value(flow_fields.fields().area) / numbers::PI),
               wall_lift,
               radius);

        weak_term(value(solid_fields.fields().displacement), test(lambda_wall))
          .add(builder);
        weak_term(negative_wall_displacement, test(multiplier)).add(builder);
        weak_term(lambda_wall, test(solid_fields.fields().velocity))
          .add(builder);

        const auto pressure      = -1. * value(multiplier);
        const auto flow_id       = flow_fields.fields().state;
        auto       pressure_term = builder.term(flow_id, "native-pressure");
        pressure_term.residual([&, pressure, flow_id, id](const auto &context) {
          auto contribution = std::make_shared<MetricFlowX::VectorType>(
            flow_problem.make_state());
          *contribution = 0.;
          flow_problem.add_external_pressure_residual(
            context.time(),
            context.state(flow_id),
            metric_flow_x_pressure_provider(pressure, context.state(id)),
            *contribution);
          dealii::PackagedOperation<MetricFlowX::VectorType> result;
          result.reinit_vector = [&flow_problem](auto &dst, const bool) {
            flow_problem.reinit_state(dst);
          };
          result.apply     = [contribution](auto &dst) { dst = *contribution; };
          result.apply_add = [contribution](auto &dst) {
            dst += *contribution;
          };
          return result;
        });
        typename std::decay_t<decltype(builder)>::Model::MatrixOperatorFactory
          pressure_jacobian = [&, pressure, flow_id](const auto &context) {
            return metric_flow_x_pressure_operator(flow_problem,
                                                   pressure,
                                                   context.state(flow_id),
                                                   context.time());
          };
        pressure_term.state(id, std::move(pressure_jacobian));

        builder.saddle_point(id,
                             {solid_fields.fields().displacement.id(),
                              solid_fields.fields().velocity.id(),
                              flow_fields.fields().state},
                             detail::make_multiplier_metric(builder,
                                                            multiplier));
        return ConstraintFields<std::decay_t<decltype(lambda)>>{multiplier};
      },
      "vessel-wall");

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
    const std::filesystem::path multiplier_output_directory =
      flow_output_directory + "/interaction";
    std::vector<std::pair<double, std::string>> multiplier_output_records;
    const auto                                  output = [&solid_problem,
                         &solid_parameters,
                         &flow_problem,
                         &adapter,
                         &lambda_view,
                         &multiplier_output_directory,
                         &multiplier_output_records,
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
      flow_problem.compute_pressure(
        flow_state,
        pressure,
        time,
        metric_flow_x_pressure_provider(
          -1. * value(coupling_fields.fields().multiplier),
          adapter.field(solution, coupling_fields.fields().multiplier)));
      flow_problem.output_results(flow_state, pressure, step);
      solid_problem.output_results();
      DataOut<1, 3> multiplier_output;
      multiplier_output.attach_dof_handler(lambda_view.dof_handler());
      multiplier_output.add_data_vector(
        adapter.field(solution, coupling_fields.fields().multiplier),
        "pressure_jump",
        DataOut<1, 3>::type_dof_data);
      multiplier_output.build_patches(lambda_view.mapping());
      const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      if (rank == 0)
        std::filesystem::create_directories(multiplier_output_directory);
      MPI_Barrier(MPI_COMM_WORLD);
      const auto filename = "pressure_jump_" + std::to_string(step) + ".vtu";
      multiplier_output.write_vtu_in_parallel(
        (multiplier_output_directory / filename).string(), MPI_COMM_WORLD);
      multiplier_output_records.emplace_back(time, filename);
      if (rank == 0)
        {
          std::ofstream pvd(multiplier_output_directory / "pressure_jump.pvd");
          DataOutBase::write_pvd_record(pvd, multiplier_output_records);
        }
      MPI_Barrier(MPI_COMM_WORLD);
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
