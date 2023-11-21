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

#include <iomanip>

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
      unsigned int                     couplingSampling         = 1;
      unsigned int                     Pa_to_dyn_conversion = 10;
      unsigned int                     couplingStart = 9;
      if (argc > 2)
        {
          prm_file        = argv[1];
          input_file_name = argv[2];
        }
      else
        prm_file = "parameters.prm";
      if (argc > 3)
        couplingSampling = std::strtol(argv[3], NULL, 10);
      if (argc > 4)
        couplingStart = std::strtol(argv[4], NULL, 10);

      if (prm_file.find("3d") != std::string::npos)
        {
          ElasticityProblemParameters<3> par;
          ElasticityProblem<3>           problem3D(par);
          ParameterAcceptor::initialize(prm_file);
          problem3D.run_timestep0();

          {
            Model1d pb1D;
            int     id = 0, p = 1;
            // define process ID and number of processes
            pb1D.partitionID = id;
            pb1D.nproc       = p;
            // initialize model
            pb1D.verbose = 0;

            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              pb1D.init(input_file_name);

            // enter time loop
            pb1D.iT         = 0; // not simple iteration counter !
            int    iter     = 0;
            double timestep = 0.0;
            double tEnd = Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.tEnd);
            double dt =
              Utilities::MPI::broadcast(MPI_COMM_WORLD, pb1D.dtMaxLTSLIMIT);

            while (timestep < tEnd)
              {
                // write files for Sarah
                pb1D.writePressure();
                pb1D.writeEXTPressure();

                // solve time step
                if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                  {
                    pb1D.solveTimeStep(pb1D.dtMaxLTSLIMIT);
                  }
                MPI_Barrier(MPI_COMM_WORLD);

                // every iterSampling we aso perform the 3D
                if (timestep > couplingStart && iter % couplingSampling == 0)
                  {
                    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                      pb1D.compute_new_displacement_for_coupling();
                    MPI_Barrier(MPI_COMM_WORLD);

                    std::vector<double> new_displacement_data =
                      Utilities::MPI::broadcast(
                        MPI_COMM_WORLD,
                        pb1D.new_displacement); // canbe omitted if all
                                                // processors execute the 1D

                    // to delete cout
                    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                      {
                        std::cout << "new displacement data: ";
                        for (long unsigned int print_index = 0;
                             print_index < pb1D.new_displacement.size();
                             ++print_index)
                          std::cout << print_index << ": " << scientific
                                    << setprecision(4)
                                    << pb1D.new_displacement[print_index]
                                    << ", ";
                        std::cout << std::endl;
                      }
                    // end cout

                    problem3D.update_inclusions_data(new_displacement_data);
                    problem3D.run_timestep();

                    // to delete cout
                    //                   if
                    //                   (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                    //                   == 0)
                    //                     std::cout << "new pressure data";
                    //                   problem3D.coupling_pressure.print(std::cout);
                    // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                    // 0)
                    // {
                    //   std::cout << "new pressure data";
                    //   for (auto print_index = 0;
                    //       print_index < problem3D.coupling_pressure.size();
                    //       ++print_index)
                    //    std::cout << print_index << ": " <<
                    //                 problem3D.coupling_pressure[print_index]
                    //                 << ", ";
                    //   std::cout << std::endl;
                    // }
                    // end cout
                    Vector<double> coupling_pressure(
                      problem3D.coupling_pressure);

                    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                      {
                        AssertDimension(problem3D.coupling_pressure.size(),
                                        pb1D.NV);
                        // to delete cout
                        std::cout << "check on applied pressure (multplied by -" << Pa_to_dyn_conversion << ")";
                        for (auto print_index = 0;
                             print_index < coupling_pressure.size();
                             ++print_index)
                          std::cout << print_index << ": "
                                    << (-10 *
                                                      coupling_pressure[print_index]) << ", ";
                        std::cout << std::endl;
                        // end cout
                        for (int i = 0; i < pb1D.NV; i++)
                          for (int j = 0; j < pb1D.vess[i].NCELLS; j++)
                            pb1D.vess[i].setpeconst(j,
                                                    (-10 *
                                                      coupling_pressure[i]));
                      }

                    for (int i = 0; i < pb1D.NV; i++)
                      {
                        if (pb1D.vess[i].bcType[0] > 1)
                          {
                            const auto iJunc      = pb1D.vess[i].iJuncL;
                            const auto iJuncOrder = pb1D.vess[i].iJuncLorder;
                            pb1D.junctionsData[iJunc].peJ[iJuncOrder] =
                              pb1D.vess[i].getpe(0, 0);
                          }
                        if (pb1D.vess[i].bcType[1] > 1)
                          {
                            const auto iJunc      = pb1D.vess[i].iJuncR;
                            const auto iJuncOrder = pb1D.vess[i].iJuncRorder;
                            pb1D.junctionsData[iJunc].peJ[iJuncOrder] =
                              pb1D.vess[i].getpe(pb1D.vess[i].NCELLS - 1,
                                                 pb1D.vess[i].nDOFs - 1);
                          }
                      }
                  }

                // write files for Sarah
                pb1D.writeArea();
                pb1D.writeFlow();

                iter++;
                pb1D.iT += 1;
                timestep += dt;
              }
            pb1D.end();
          }
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
