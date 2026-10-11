// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the ImmersX authors
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_explicit.h>

#include <string>

using namespace ImmersX;

int
main(int argc, char *argv[])
{
  using namespace dealii;
  deallog.depth_console(1);
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      deallog.depth_console(
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ? 1 : 0);

      const std::string prm_file =
        argc > 1 ? argv[1] : "prms/prm_2d_euler_mms.prm";
      const auto dimensions = get_dimension_parameters(prm_file);

      if (dimensions.dimension == 2 && dimensions.space_dimension == 3)
        {
          ElasticityProblemParameters<2, 3> parameters;
          ElasticityExplicit<2, 3>          problem(parameters);
          initialize_parameters(prm_file);
          problem.run();
        }
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        {
          ElasticityProblemParameters<3> parameters;
          ElasticityExplicit<3>          problem(parameters);
          initialize_parameters(prm_file);
          problem.run();
        }
      else if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        {
          ElasticityProblemParameters<2> parameters;
          ElasticityExplicit<2>          problem(parameters);
          initialize_parameters(prm_file);
          problem.run();
        }
      else
        throw_unsupported_dimension_combination(dimensions);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
