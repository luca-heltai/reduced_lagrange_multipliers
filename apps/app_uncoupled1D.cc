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
#include "coupledModel1d.h"

int
main(int argc, char *argv[])
{
  using namespace dealii;
  deallog.depth_console(1);
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      input_file_name;
      unsigned int                     couplingSampling      = 1;
      unsigned int                     couplingStart         = 9;

      if (argc > 1)
      {
        input_file_name = argv[1];
        if (argc > 3)
        {
          couplingSampling = std::strtol(argv[2], NULL, 10);
          couplingStart    = std::strtol(argv[3], NULL, 10);
        }

        std::cout << "Running 1d problem" << std::endl;

          {
            CoupledModel1d pb1D;
            int            id = 0, p = 1;
            // define process ID and number of processes
            pb1D.partitionID = id;
            pb1D.nproc       = p;
            // initialize model
            pb1D.verbose = 0;
            pb1D.init(input_file_name);

            // enter time loop
            pb1D.iT         = 0; // not simple iteration counter !
            int    iter     = 0;
            double timestep = 0.0;
            double tEnd = pb1D.tEnd;
            double dt = pb1D.dtMaxLTSLIMIT;

            while (timestep < tEnd)
              {
                // solve time step

                    // write files for Sarah
                    pb1D.writePressure();
                    pb1D.writeEXTPressure();
                    pb1D.solveTimeStep(pb1D.dtMaxLTSLIMIT);

                // every iterSampling we aso perform the 3D
                if (timestep > couplingStart && iter % couplingSampling == 0)
                {

                  pb1D.solvePseudo3D();

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
                pb1D.closeFilesPlot();
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
