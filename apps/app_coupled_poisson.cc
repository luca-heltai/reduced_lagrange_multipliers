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

#include <deal.II/numerics/data_out.h>

#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
  class CoupledPoissonApplicationParameters : public dealii::ParameterAcceptor
  {
  public:
    CoupledPoissonApplicationParameters()
      : ParameterAcceptor("/Coupled Poisson/")
    {
      add_parameter("Output directory", output_directory);
      add_parameter("Multiplier output name", multiplier_output_name);
    }

    std::string output_directory       = ".";
    std::string multiplier_output_name = "multiplier";
  };

  void
  run_coupled_poisson(const std::string &parameter_file)
  {
    using namespace ImmersX;

    CoupledPoissonApplicationParameters application_parameters;
    PoissonParameters<2>                bulk_parameters;
    PoissonParameters<1, 2> embedded_parameters("/Embedded Poisson/");
    LinearSolverParameters  adapter_parameters;
    FiniteElementSpaceParameters<1, 2> multiplier_parameters(
      "/Coupled Poisson/Multiplier space/", "FE_DGQ<1>(0)");
    initialize_parameters(parameter_file);

    bulk_parameters.output_directory = application_parameters.output_directory;
    embedded_parameters.output_directory =
      application_parameters.output_directory;

    PoissonSolver<2>    bulk_problem(bulk_parameters);
    PoissonSolver<1, 2> embedded_problem(embedded_parameters);

    const auto initialize_problem = [](auto &problem) {
      problem.make_grid();
      problem.setup_fe();
      problem.setup_system();
      problem.assemble_system();
    };
    initialize_problem(bulk_problem);
    initialize_problem(embedded_problem);

    using FieldVector  = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

    Adapter    adapter(adapter_parameters, MPI_COMM_WORLD);
    const auto bulk     = adapter.add(bulk_problem, "bulk");
    const auto embedded = adapter.add(embedded_problem, "embedded");

    // Independent multiplier DoFs share the embedded geometry.
    FiniteElementSpace<1, 2> multiplier_space(embedded_problem.triangulation(),
                                              multiplier_parameters);
    AssertThrow(multiplier_space.finite_element().n_components() == 1,
                ExcMessage("The multiplier FE must be scalar."));
    const auto multiplier_view = multiplier_space.view();
    const auto bulk_field      = bulk.fields().solution;
    const auto embedded_field  = embedded.fields().solution;
    const auto multiplier      = multiplier_view.field("lambda");
    const auto constraint =
      make_continuity_constraint(bulk_field, embedded_field, multiplier);
    const auto coupling = adapter.add(constraint, "continuity");

    auto state = adapter.make_state();
    adapter.solve(state);

    bulk_problem.set_solution(adapter.field(state, bulk.fields().solution));
    embedded_problem.set_solution(
      adapter.field(state, embedded.fields().solution));

    bulk_problem.output_results();
    embedded_problem.output_results();

    std::filesystem::create_directories(
      application_parameters.output_directory);
    dealii::DataOut<1, 2> multiplier_output;
    multiplier_output.attach_dof_handler(multiplier_space.dof_handler());
    multiplier_output.add_data_vector(
      adapter.field(state, coupling.fields().multiplier),
      "lagrange_multiplier",
      dealii::DataOut<1, 2>::type_dof_data);
    multiplier_output.build_patches();
    const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto multiplier_filename =
      std::filesystem::path(application_parameters.output_directory) /
      (application_parameters.multiplier_output_name + "-0." +
       std::to_string(rank) + ".vtu");
    std::ofstream multiplier_file(multiplier_filename);
    multiplier_output.write_vtu(multiplier_file);
    if (rank == 0)
      {
        std::ofstream multiplier_pvd(
          std::filesystem::path(application_parameters.output_directory) /
          (application_parameters.multiplier_output_name + ".pvd"));
        dealii::DataOutBase::write_pvd_record(
          multiplier_pvd,
          {{0., application_parameters.multiplier_output_name + "-0.0.vtu"}});
      }

    auto residual = adapter.make_state();
    adapter.evaluate_residual(state, residual);
    const double residual_norm = residual.l2_norm();
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "multiplier_dofs = "
                << multiplier_space.dof_handler().n_dofs() << '\n'
                << "coupled_residual = " << residual_norm << '\n';

    AssertThrow(std::isfinite(residual_norm) && residual_norm < 1.e-7,
                dealii::ExcMessage(
                  "The coupled Poisson residual is too large."));
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
      const std::string parameter_file = argc > 1 ? argv[1] : "parameters.prm";
      const auto dimensions = ImmersX::get_dimension_parameters(parameter_file);

      if (dimensions.dimension != 2 || dimensions.space_dimension != 2 ||
          dimensions.reduced_dimension != 1)
        ImmersX::throw_unsupported_dimension_combination(dimensions);

      run_coupled_poisson(parameter_file);
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << std::endl;
      return 1;
    }

  return 0;
}
