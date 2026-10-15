// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <immersx/io/utils.h>
#include <immersx/physics/lifted_poisson.h>

int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
      ImmersX::LiftedPoissonParameters         parameters;
      ImmersX::initialize_parameters(argc > 1 ? argv[1] : "lifted_poisson.prm");
      ImmersX::run_lifted_poisson(parameters);
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << std::endl;
      return 1;
    }
  return 0;
}
