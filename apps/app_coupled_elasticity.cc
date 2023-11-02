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
#include "model1d.h"
int
main(int argc, char *argv[])
{
  using namespace dealii;
  deallog.depth_console(1);
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      prm_file;
      std::string                      input_file_name;
      unsigned int                     iterSampling = 100;
      // unsigned int                     Pa_to_dyn_conversion = 10;
      if (argc > 2)
        {
          prm_file        = argv[1];
          input_file_name = argv[2];
        }
      if (argc > 3)
        iterSampling = std::strtol(argv[3], NULL, 10);
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

          {
            Model1d pb1D;
            int id = 0, p = 1;
            // define process ID and number of processes
            pb1D.partitionID = id;
            pb1D.nproc       = p;
            // initialize model
            pb1D.verbose = 0;
            pb1D.init(input_file_name);

            // enter time loop
            pb1D.iT = 0; // not simple iteration counter !
            int iter = 0;
            double timestep = 0.0;

            while(timestep < pb1D.tEnd)
              {
                // write files for Sarah
		            pb1D.writePressure();
            		pb1D.writeEXTPressure();
                // solve time step
                pb1D.solveTimeStep(pb1D.dtMaxLTSLIMIT);

                if (iter > 0 && iter % iterSampling == 0)
                {
                  // non fare il 3d a tutti tutti i timestep
                  pb1D.compute_new_displacement_for_coupling();
                  problem3D.update_inclusions_data(pb1D.new_displacement);
                  
                  std::cout << "new displacement data: ";
                  for (auto print_index = 0; print_index < pb1D.new_displacement.size(); ++print_index )
                    // std::cout << print_index << ": " << pb1D.new_displacement[print_index] << ", ";
                    std::cout << print_index << ": " << scientific << setprecision(4) << pb1D.new_displacement[print_index] << ", ";
                  std::cout << std::endl;

                  problem3D.run_timestep();

                  std::cout << "new pressure data";
                  for (auto print_index = 0; print_index < problem3D.coupling_pressure.size(); ++print_index )
                    // std::cout << print_index << ": " << problem3D.coupling_pressure[print_index] << ", ";
                    std::cout << print_index << ": " << scientific << setprecision(4) << problem3D.coupling_pressure[print_index] << ", ";
                  std::cout << std::endl;

                  AssertDimension(problem3D.coupling_pressure.size(), pb1D.NV);

                  for (int i = 0; i < pb1D.NV; i++)
                    for (int j = 0; j < pb1D.vess[i].NCELLS; j++)
                      pb1D.vess[i].setpeconst(j, (-problem3D.coupling_pressure[i]));
                }
                
                // write files for Sarah
            		pb1D.writeArea();
            		pb1D.writeFlow();
            
                iter++;
                pb1D.iT += 1;
                timestep += pb1D.dtMaxLTSLIMIT;
              }
            pb1D.end();
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
