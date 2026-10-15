// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#include <deal.II/base/mpi.h>

#include <immersx/physics/lifted_elasticity.h>

namespace
{
  template <int dim>
  void
  run(const std::string &filename)
  {
    ImmersX::LiftedElasticityParameters<dim> parameters;
    ImmersX::initialize_parameters(filename);
    ImmersX::LiftedElasticity<dim> problem(parameters);
    problem.run();
  }
} // namespace

int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
      const std::string filename = argc > 1 ? argv[1] : "lifted_elasticity.prm";
      const auto dimensions      = ImmersX::get_dimension_parameters(filename);
      if (dimensions.dimension == 2 && dimensions.space_dimension == 2)
        run<2>(filename);
      else if (dimensions.dimension == 3 && dimensions.space_dimension == 3)
        run<3>(filename);
      else
        ImmersX::throw_unsupported_dimension_combination(dimensions);
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << std::endl;
      return 1;
    }
}
