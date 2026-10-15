// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/config.h>

#include <gtest/gtest.h>
#ifdef DEAL_II_WITH_VTK
#  include <immersx/physics/lifted_poisson.h>

#  include <sstream>

#  include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  void
  adaptive_output(bool default_modes, bool imported_data = false)
  {
    if (imported_data && !SymbolicFieldEvaluator::available())
      GTEST_SKIP() << "Imported modal expressions require SymEngine.";
    ParameterAcceptor::clear();
    LiftedPoissonParameters parameters;
    ParameterAcceptor::initialize(
      TestPaths::parameter_path("gtests/parameters/lifted_poisson.prm"));
    if (default_modes)
      parameters.lift.section.selected_coefficients.clear();
    if (imported_data)
      {
        parameters.modal_data_fields = "r=cell:radius,h=point:height";
        parameters.modal_data        = "1+r*r+0.01*h;0.1*r;0.2*r+0*pi+t";
      }
    parameters.output.output_directory = TestPaths::output_directory(
      std::string("lifted-poisson-application-") +
      (default_modes ? "default-" :
                       (imported_data ? "imported-" : "selected-")) +
      std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
    run_lifted_poisson(parameters);
    const auto directory =
      std::filesystem::path(parameters.output.output_directory);
    for (unsigned int cycle = 0; cycle < parameters.refinement_cycles; ++cycle)
      {
        EXPECT_TRUE(std::filesystem::is_regular_file(
          directory / ("bulk_" + std::to_string(cycle) + ".pvd")));
        EXPECT_TRUE(std::filesystem::is_regular_file(
          directory / ("modes_" + std::to_string(cycle) + ".pvd")));
        const auto filename =
          directory / ("lifted_" + std::to_string(cycle) + ".vtu");
        ASSERT_TRUE(std::filesystem::is_regular_file(filename));
        std::ifstream     input(filename);
        const std::string text((std::istreambuf_iterator<char>(input)), {});
        EXPECT_NE(text.find("Name=\"lambda\""), std::string::npos);
        EXPECT_NE(text.find("Name=\"weight\""), std::string::npos);
      }
    std::ifstream input(directory / "diagnostics.csv");
    ASSERT_TRUE(input.good());
    std::string line;
    std::getline(input, line);
    EXPECT_EQ(
      line,
      "cycle,cells,bulk_dofs,modal_dofs,residual,bulk_norm,modal_norm,L2_error,H1_error");
    unsigned int count          = 0;
    double       previous_cells = 0.;
    while (std::getline(input, line))
      {
        std::istringstream  stream(line);
        std::vector<double> values;
        std::string         column;
        while (std::getline(stream, column, ','))
          values.push_back(std::stod(column));
        ASSERT_EQ(values.size(), 9u);
        for (const double value : values)
          EXPECT_TRUE(std::isfinite(value));
        EXPECT_EQ(values[0], count++);
        EXPECT_GT(values[1], previous_cells);
        previous_cells = values[1];
        EXPECT_LT(values[4], 1.e-7);
        EXPECT_GT(values[5], 1.e-3);
        EXPECT_GT(values[6], 1.e-3);
        EXPECT_GT(values[7], 0.);
        EXPECT_GT(values[8], 0.);
      }
    EXPECT_EQ(count, parameters.refinement_cycles);
  }
} // namespace
TEST(LiftedPoissonApplication, BOTH_AdaptiveOutputAndDiagnostics)
{
  adaptive_output(false);
}
TEST(LiftedPoissonApplication, BOTH_DefaultSelectedModes)
{
  adaptive_output(true);
}
TEST(LiftedPoissonApplication, BOTH_ImportedModalDataAndAliases)
{
  adaptive_output(false, true);
}
#endif
