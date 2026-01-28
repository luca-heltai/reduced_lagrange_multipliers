// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include "blood_flow_1d.h"

using namespace dealii;

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      deallog.depth_console(0); // Reduce deallog output

      std::string prm_file;
      if (argc > 1)
        prm_file = argv[1];
      else
        prm_file = "blood_flow_parameters.prm";

      // 1D vessels embedded in 3D space
      BloodFlowParameters parameters;
      ParameterAcceptor::initialize(prm_file);

      // Create output directory if it doesn't exist
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          const std::string command = "mkdir -p " + parameters.output_directory;
          const int         result  = std::system(command.c_str());
          (void)result; // Suppress unused variable warning
        }

      BloodFlow1D<3> blood_flow_problem(parameters);
      blood_flow_problem.run();
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
