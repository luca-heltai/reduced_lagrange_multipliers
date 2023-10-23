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
      std::string			input_file_name;
      if (argc > 2)
      {
      	prm_file = argv[1];
      	input_file_name = argv[2];
      }
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
          Model1d pb;
          pb.init(input_file_name);
          
          ElasticityProblemParameters<3> par;
          ElasticityProblem<3>           problem(par);
          ParameterAcceptor::initialize(prm_file);
          problem.run_timestep0();
          
          {
          	int id = 0, p = 1;
	// define process ID and number of processes
	pb.partitionID = id;
	pb.nproc = p;

	// initialize model
	pb.verbose = 0;

	pb.init(input_file_name);

	// enter time loop
	pb.iT = 0;
	//double peConst = 0.;

	//pressure read from file
	//pb.ReadPressure();


	double oneDimArea[pb.NV];
	double threeDimPressure[pb.NV];

	for(int iter = 0; iter < 10000000000000; iter++){
		// solve time step
		pb.solveTimeStep(pb.dtMaxLTSLIMIT);
		// get pressure and diameter along vessel
		if(pb.sampleSpace==1){
			double qSample[10];
			for(int i = 0; i< pb.NV; i++){
				for(int j = 0; j< pb.vess[i].solutionSpaceX.size(); j++){
					pb.sampleMid(i,qSample,pb.vess[i].solutionSpaceX[j]);
					pb.vess[i].solutionSpaceP[j] = qSample[3];
					pb.vess[i].solutionSpaceD[j] = 2.*sqrt(qSample[0]/M_PI);
				}
			}
		}

		// pass 1D state to 3D solid model
		for(int i = 0; i< pb.NV; i++){
			double qSample[10];
			pb.sampleMid(i,qSample,pb.vess[i].L/2.);
			oneDimArea[i] = qSample[0];
		}
		// solve 3D solid model
		// pass external pressure
		std::vector<double> new_data(3, 0.0);
		problem.update_inclusions_data(new_data);
		         problem.run_timestep();
		         
		// loop over vessels
		
	//	for(int i = 0; i< pb.NV; i++){

	//		threeDimPressure[i] = pb.ExternalPressure[i]; // TO BE REPLACED
	//	}
		// set external pressure for 1D model
		if(pb.setExternalPressure){
			for(int i = 0; i< pb.NV; i++){
				for(int j = 0; j< pb.vess[i].NCELLS; j++){
					pb.vess[i].setpeconst(j, threeDimPressure[i]);

				}
			}
		}

		pb.iT += 1;
		if (pb.endOfSimulation==1)
			break;
	}
          }
                    pb.end();

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
