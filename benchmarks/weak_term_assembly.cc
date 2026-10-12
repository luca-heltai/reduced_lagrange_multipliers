// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_creator.h>

#include <immersx/core/fe_space.h>
#include <immersx/core/state.h>
#include <immersx/core/weak_term.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;
using namespace ImmersX;

namespace
{
  using Clock = std::chrono::steady_clock;

  double
  milliseconds(const Clock::duration duration)
  {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  std::pair<double, double>
  median_and_mad(std::vector<double> samples)
  {
    std::sort(samples.begin(), samples.end());
    const auto median = samples[samples.size() / 2];
    for (auto &sample : samples)
      sample = std::abs(sample - median);
    std::sort(samples.begin(), samples.end());
    return {median, samples[samples.size() / 2]};
  }

  template <typename MatrixA, typename MatrixB>
  double
  relative_matrix_difference(const MatrixA &a, const MatrixB &b)
  {
    AssertDimension(a.m(), b.m());
    AssertDimension(a.n(), b.n());
    double difference_squared = 0.;
    double reference_squared  = 0.;
    for (unsigned int i = 0; i < a.m(); ++i)
      for (unsigned int j = 0; j < a.n(); ++j)
        {
          const double av = a.el(i, j);
          const double bv = b.el(i, j);
          difference_squared += (av - bv) * (av - bv);
          reference_squared += bv * bv;
        }
    return std::sqrt(difference_squared / std::max(reference_squared, 1.e-300));
  }

  unsigned int
  argument(const int          argc,
           char             **argv,
           const std::string &name,
           const unsigned int fallback)
  {
    const std::string prefix = "--" + name + "=";
    for (int i = 1; i < argc; ++i)
      {
        const std::string item(argv[i]);
        if (item.rfind(prefix, 0) == 0)
          return static_cast<unsigned int>(
            std::stoul(item.substr(prefix.size())));
      }
    return fallback;
  }
} // namespace

int
main(int argc, char **argv)
{
  const auto refinements = argument(argc, argv, "refinements", 4);
  const auto repetitions = argument(argc, argv, "repetitions", 9);
  AssertThrow(repetitions > 0,
              ExcMessage("The benchmark needs at least one repetition."));

  const auto       setup_start = Clock::now();
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation, -1., 1.);
  triangulation.refine_global(refinements);
  FE_Q<2>                   finite_element(1);
  DoFHandler<2>             dof_handler(triangulation);
  AffineConstraints<double> constraints;
  dof_handler.distribute_dofs(finite_element);
  constraints.close();

  StateLayout layout;
  const auto  space =
    fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);
  const auto source            = space.field(layout, "source");
  const auto target            = space.field(layout, "target");
  const auto source_expression = value(source);
  const auto test_expression   = test(target);
  const auto common_setup_ms   = milliseconds(Clock::now() - setup_start);
  using Assembly = detail::WeakAssembly<decltype(source_expression),
                                        decltype(test_expression)>;
  using Vector   = dealii::Vector<double>;
  using Matrix   = dealii::SparseMatrix<double>;

  const auto immersed_assembly = [&] {
    return Assembly::template assemble<Vector, Matrix>(source_expression,
                                                       test_expression);
  };

  const auto deal_ii_assembly = [&] {
    DynamicSparsityPattern dynamic_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dynamic_sparsity,
                                    constraints,
                                    false);
    auto sparsity = std::make_shared<SparsityPattern>();
    sparsity->copy_from(dynamic_sparsity);
    auto matrix = std::make_shared<Matrix>(*sparsity);
    MatrixCreator::create_mass_matrix(StaticMappingQ1<2>::mapping,
                                      dof_handler,
                                      QGauss<2>(finite_element.degree + 1),
                                      *matrix,
                                      static_cast<const Function<2, double> *>(
                                        nullptr),
                                      constraints);
    return std::make_pair(std::move(matrix), std::move(sparsity));
  };

  const auto cold_immersx_start = Clock::now();
  auto       immersed           = immersed_assembly();
  const auto cold_immersx_ms = milliseconds(Clock::now() - cold_immersx_start);

  const auto cold_dealii_start = Clock::now();
  auto       native            = deal_ii_assembly();
  const auto cold_dealii_ms    = milliseconds(Clock::now() - cold_dealii_start);

  const double relative_error =
    relative_matrix_difference(*immersed.matrix, *native.first);
  AssertThrow(relative_error < 1.e-12,
              ExcMessage("ImmersX and MatrixCreator mass matrices differ: " +
                         std::to_string(relative_error)));

  // Exclude one untimed warm-up from the repeated samples.
  immersed = immersed_assembly();
  native   = deal_ii_assembly();
  std::vector<double> immersed_samples;
  std::vector<double> native_samples;
  immersed_samples.reserve(repetitions);
  native_samples.reserve(repetitions);
  for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
    {
      if (repetition % 2 == 0)
        {
          const auto immersed_start = Clock::now();
          immersed                  = immersed_assembly();
          immersed_samples.push_back(
            milliseconds(Clock::now() - immersed_start));

          const auto native_start = Clock::now();
          native                  = deal_ii_assembly();
          native_samples.push_back(milliseconds(Clock::now() - native_start));
        }
      else
        {
          const auto native_start = Clock::now();
          native                  = deal_ii_assembly();
          native_samples.push_back(milliseconds(Clock::now() - native_start));

          const auto immersed_start = Clock::now();
          immersed                  = immersed_assembly();
          immersed_samples.push_back(
            milliseconds(Clock::now() - immersed_start));
        }
    }

  const auto [immersed_median, immersed_mad] = median_and_mad(immersed_samples);
  const auto [native_median, native_mad]     = median_and_mad(native_samples);
  std::cout << "{\"case\":\"mass_2d_q1\",\"deal_ii_version\":\""
            << DEAL_II_PACKAGE_VERSION
            << "\",\"ranks\":1,\"dofs\":" << dof_handler.n_dofs()
            << ",\"cells\":" << triangulation.n_active_cells()
            << ",\"quadrature_points_per_cell\":4"
            << ",\"common_setup_ms\":" << common_setup_ms
            << ",\"relative_matrix_error\":" << relative_error
            << ",\"cold_immersx_ms\":" << cold_immersx_ms
            << ",\"cold_dealii_ms\":" << cold_dealii_ms
            << ",\"cold_ratio\":" << cold_immersx_ms / cold_dealii_ms
            << ",\"warm_immersx_median_ms\":" << immersed_median
            << ",\"warm_immersx_mad_ms\":" << immersed_mad
            << ",\"warm_dealii_median_ms\":" << native_median
            << ",\"warm_dealii_mad_ms\":" << native_mad
            << ",\"warm_ratio\":" << immersed_median / native_median
            << ",\"repetitions\":" << repetitions << "}\n";
}
