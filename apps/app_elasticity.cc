/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 * Modified by: Luca Heltai, 2020
 */

#include "elasticity.h"
int
main(int argc, char *argv[])
{
  using namespace dealii;
  deallog.depth_console(1);
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      prm_file;
      int                     kPa_to_dyn_conversion = 10;

      if (argc > 1)
        prm_file = argv[1];
      else
        prm_file = "parameters.prm";
      if (prm_file.find("23d") != std::string::npos)
        {
          ElasticityProblemParameters<2, 3> par;
          ElasticityProblem<2, 3>           problem(par);
          ParameterAcceptor::initialize(prm_file);
          problem.run();
        }
      else if (prm_file.find("3d") != std::string::npos)
        {
          ElasticityProblemParameters<3> par;
          ElasticityProblem<3>           problem3D(par);
          ParameterAcceptor::initialize(prm_file);
          problem3D.run_timestep0();
          problem3D.run_timestep();
          Vector<double> coupling_pressure(
                      problem3D.coupling_pressure);
          coupling_pressure *= kPa_to_dyn_conversion;
          coupling_pressure *= -1;


          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              // to delete cout
              std::cout << "check on applied pressure (multiplied by -" << kPa_to_dyn_conversion << ")";
              for (auto print_index = 0;
                   print_index < coupling_pressure.size();
                   ++print_index)
                std::cout << print_index << ": "
                          << (coupling_pressure[print_index]) << ", ";
              std::cout << std::endl;
            }
        }
      else
        {
          ElasticityProblemParameters<2> par;
          ElasticityProblem<2>           problem(par);
          ParameterAcceptor::initialize(prm_file);
          problem.run();
        }
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
