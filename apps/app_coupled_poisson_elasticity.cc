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

#include <immersx/core/linear_adapter.h>
#include <immersx/core/observable.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>
#include <immersx/io/output_handler.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <iostream>

#include "coupled_poisson_elasticity.h"

using namespace ImmersX;

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);

      PoissonParameters<1, 3>       poisson_parameters;
      ElasticStaticParameters<3, 3> elasticity_parameters;
      TensorProductLift<1, 2, 3>    pressure_lift("/Pressure lift/");
      CoupledPoissonElasticity::CylinderSurface surface;
      LinearSolverParameters                    adapter_parameters;
      const std::string prm_file = argc > 1 ? argv[1] : "parameters.prm";
      initialize_parameters(prm_file);

      PoissonSolver<1, 3> poisson_problem(poisson_parameters);
      poisson_problem.setup();

      ElasticStaticProblem<3, 3> elasticity_problem(elasticity_parameters);
      elasticity_problem.setup();

      using FieldVector  = ImmersXLA::MPI::Vector;
      using GlobalVector = ImmersXLA::MPI::BlockVector;
      using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

      Adapter adapter(adapter_parameters, MPI_COMM_WORLD);

      const auto poisson = adapter.add(poisson_problem).fields();
      const auto elastic = adapter.add(elasticity_problem).fields();
      const auto pressure =
        make_lift(2. * value(poisson.solution), pressure_lift);
      adapter.add(weak_term(pressure * normal(surface),
                            test(elastic.displacement)),
                  "pressure-traction");

      auto state = adapter.make_state();
      adapter.solve(state);

      OutputHandler<1, 3, FieldVector> pressure_output(
        poisson.solution,
        poisson_parameters.output_directory,
        poisson_parameters.output_name);
      OutputHandler<3, 3, FieldVector> displacement_output(
        elastic.displacement,
        elasticity_parameters.output_directory,
        elasticity_parameters.output_name);
      adapter.write_output(pressure_output, state);
      adapter.write_output(displacement_output, state);
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << std::endl;
      return 1;
    }

  return 0;
}
