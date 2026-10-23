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

#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/output_handler.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

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

    bulk_problem.setup();
    embedded_problem.setup();

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
    const auto multiplier      = multiplier_view.field("lagrange_multiplier");
    const auto constraint =
      make_continuity_constraint(bulk_field, embedded_field, multiplier);
    const auto coupling = adapter.add(constraint, "continuity");

    auto state = adapter.make_state();
    adapter.solve(state);

    OutputHandler<2, 2, FieldVector> bulk_output(
      bulk_field,
      application_parameters.output_directory,
      bulk_parameters.output_name);
    OutputHandler<1, 2, FieldVector> embedded_output(
      embedded_field,
      application_parameters.output_directory,
      embedded_parameters.output_name);
    OutputHandler<1, 2, FieldVector> multiplier_output(
      coupling.fields().multiplier,
      application_parameters.output_directory,
      application_parameters.multiplier_output_name);
    adapter.write_output(bulk_output, state);
    adapter.write_output(embedded_output, state);
    adapter.write_output(multiplier_output, state);
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "multiplier_dofs = "
                << multiplier_space.dof_handler().n_dofs() << '\n';
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
