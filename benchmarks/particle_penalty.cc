// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2026 by the ImmersX authors
#include <deal.II/base/mpi.h>

#include <immersx/core/fe_space.h>
#include <immersx/core/weak_term.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "native_particle_penalty.h"

namespace
{
  using namespace dealii;
  using Clock  = std::chrono::steady_clock;
  using Matrix = NativeParticlePenalty::Matrix;
  using Vec    = NativeParticlePenalty::Vector;

  unsigned int
  argument(int                argc,
           char             **argv,
           const std::string &name,
           unsigned int       fallback)
  {
    for (int i = 1; i < argc; ++i)
      {
        const std::string arg(argv[i]);
        const std::string prefix = "--" + name + "=";
        if (arg.rfind(prefix, 0) == 0)
          return std::stoul(arg.substr(prefix.size()));
      }
    return fallback;
  }

  std::pair<double, double>
  median_mad(std::vector<double> samples)
  {
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    for (auto &s : samples)
      s = std::abs(s - median);
    std::sort(samples.begin(), samples.end());
    return {median, samples[samples.size() / 2]};
  }

  std::pair<double, double>
  errors(const Matrix &matrix, const Matrix &reference)
  {
    AssertDimension(matrix.m(), reference.m());
    AssertDimension(matrix.n(), reference.n());
    double difference = 0, norm = 0, maximum = 0;
    // Check every entry, including entries absent from either sparsity pattern.
    for (auto i = reference.local_range().first;
         i < reference.local_range().second;
         ++i)
      for (unsigned int j = 0; j < reference.n(); ++j)
        {
          const double delta = matrix.el(i, j) - reference.el(i, j);
          maximum            = std::max(maximum, std::abs(delta));
          difference += delta * delta;
          norm += reference.el(i, j) * reference.el(i, j);
        }
    difference = Utilities::MPI::sum(difference, MPI_COMM_WORLD);
    norm       = Utilities::MPI::sum(norm, MPI_COMM_WORLD);
    maximum    = Utilities::MPI::max(maximum, MPI_COMM_WORLD);
    AssertThrow(norm > 0, ExcMessage("Penalty oracle is a zero matrix."));
    return {std::sqrt(difference / norm), maximum};
  }

  template <int support_dim, int spacedim, bool vector_valued>
  void
  benchmark(const unsigned int refinements, const unsigned int repetitions)
  {
    NativeParticlePenalty::Fixture<support_dim, spacedim, vector_valued>
           fixture(refinements);
    Matrix manual, native;
    // Establish the oracle before constructing any ImmersX object.
    fixture.explicit_particles(manual);
    fixture.coupling_matrix(native);
    const auto [native_error, native_max_error] = errors(native, manual);
    AssertThrow(
      native_error < 1.e-12 && native_max_error < 1.e-12,
      ExcMessage(
        "DoFHandlerCoupling differs from explicit particle assembly."));

    const auto           space = ImmersX::fe_space(fixture.dh,
                                         StaticMappingQ1<spacedim>::mapping,
                                         fixture.constraints,
                                         fixture.relevant);
    ImmersX::StateLayout layout;
    const auto           u = [&]() {
      if constexpr (vector_valued)
        return ImmersX::vector_field(space, layout, "u");
      else
        return ImmersX::scalar_field(space, layout, "u");
    }();
    const ImmersX::IntegrationSupport<support_dim, spacedim> support(
      fixture.immersed);
    using Model = ImmersX::SemiDiscreteModel<Vec, Matrix>;
    std::shared_ptr<Model> model;
    const auto             assemble = [&]() {
      model = std::make_shared<Model>();
      ImmersX::SemidiscreteBuilder<Vec, Matrix> builder(layout, *model);
      ImmersX::weak_term(fixture.penalty * ImmersX::value(u),
                         ImmersX::test(u),
                         support)
        .add(builder);
      const ImmersX::StateView<Vec>         state(layout, 0.);
      const ImmersX::EvaluationContext<Vec> context(0., state);
      const auto                            matrix =
        model->state_matrix_operator(u.field_id(), u.field_id(), context);
      AssertThrow(matrix.has_value() && matrix->matrix() != nullptr,
                  ExcMessage("ImmersX penalty matrix was not materialized."));
      return matrix->matrix();
    };
    const auto matrix                    = assemble();
    const auto [matrix_error, max_error] = errors(*matrix, native);
    Vec input, native_action, action, constant;
    for (auto *v : {&input, &native_action, &action, &constant})
      v->reinit(fixture.dh.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto i : fixture.dh.locally_owned_dofs())
      {
        input[i]    = std::sin(0.37 * (i + 1)) + 0.25;
        constant[i] = fixture.constraints.is_constrained(i) ? 0. : 1.;
      }
    input.compress(VectorOperation::insert);
    constant.compress(VectorOperation::insert);
    fixture.constraints.distribute(input);
    matrix->vmult(action, input);
    native.vmult(native_action, input);
    action -= native_action;
    const double action_error = action.l2_norm() / native_action.l2_norm();
    AssertThrow(matrix_error < 1.e-12 && max_error < 1.e-12 &&
                  action_error < 1.e-12,
                ExcMessage("ImmersX differs from the step-80 penalty matrix."));

    matrix->vmult(action, constant);
    const double expected =
      fixture.penalty * fixture.measure * fixture.fe.n_components();
    const double measure_error =
      std::abs(constant * action - expected) / expected;
    AssertThrow(
      measure_error < 1.e-12,
      ExcMessage(
        "Distributed assembly lost or duplicated integration measure."));

    // Both public paths rebuild/search particles and materialize a complete
    // matrix. Native starts from common bulk sparsity; ImmersX builds its own.
    std::vector<double> native_samples, immersed_samples;
    const auto          sample = [](auto &&f, auto &samples) {
      MPI_Barrier(MPI_COMM_WORLD);
      const auto start = Clock::now();
      f();
      const double elapsed =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      samples.push_back(Utilities::MPI::max(elapsed, MPI_COMM_WORLD));
    };
    fixture.coupling_matrix(native);
    assemble();
    for (unsigned int r = 0; r < repetitions; ++r)
      if (r % 2 == 0)
        {
          sample([&]() { fixture.coupling_matrix(native); }, native_samples);
          sample(assemble, immersed_samples);
        }
      else
        {
          sample(assemble, immersed_samples);
          sample([&]() { fixture.coupling_matrix(native); }, native_samples);
        }
    const auto [im, iad] = median_mad(immersed_samples);
    const auto [nm, nad] = median_mad(native_samples);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return;
    std::cout
      << std::setprecision(17) << "{\"case\":\"particle_penalty_" << support_dim
      << "d_in_" << spacedim << "d_" << (vector_valued ? "vector" : "scalar")
      << "\","
      << "\"deal_ii_version\":\"" << DEAL_II_PACKAGE_VERSION << "\","
      << "\"dimension\":" << spacedim
      << ",\"support_dimension\":" << support_dim
      << ",\"components\":" << fixture.fe.n_components()
      << ",\"ranks\":" << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
      << ",\"active_particle_ranks\":" << fixture.active_particle_ranks
      << ",\"dofs\":" << fixture.dh.n_dofs()
      << ",\"cells\":" << fixture.background.n_global_active_cells()
      << ",\"support_cells\":" << fixture.immersed.n_global_active_cells()
      << ",\"quadrature_points_per_cell\":" << fixture.quadrature.size()
      << ",\"refinements\":" << refinements
      << ",\"repetitions\":" << repetitions
      << ",\"penalty\":" << fixture.penalty
      << ",\"relative_native_manual_error\":" << native_error
      << ",\"native_max_entry_error\":" << native_max_error
      << ",\"relative_matrix_error\":" << matrix_error
      << ",\"max_entry_error\":" << max_error
      << ",\"relative_action_error\":" << action_error
      << ",\"constant_measure_error\":" << measure_error
      << ",\"warm_immersx_median_ms\":" << im
      << ",\"warm_immersx_mad_ms\":" << iad
      << ",\"warm_dealii_median_ms\":" << nm
      << ",\"warm_dealii_mad_ms\":" << nad << ",\"warm_ratio\":" << im / nm
      << ",\"timing_equivalent\":true"
      << ",\"timing_scope\":\"distributed_particle_generation_search_and_materialized_matrix_max_rank\""
      << ",\"native_reference\":\"step-80 DoFHandlerCoupling::create_nitsche_restriction_matrix\""
      << ",\"upstream_sha\":\"c061c24bd41b4eb9fb62d1625dc0b818a54bc2bd\"}\n";
  }
} // namespace

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const auto refinements = argument(argc, argv, "refinements", 2);
  const auto repetitions = argument(argc, argv, "repetitions", 9);
  AssertThrow(repetitions > 0 && refinements <= 4,
              ExcMessage(
                "Use positive repetitions and at most four refinements."));
  benchmark<1, 2, false>(refinements, repetitions);
  benchmark<1, 2, true>(refinements, repetitions);
  benchmark<2, 3, false>(refinements, repetitions);
  benchmark<2, 3, true>(refinements, repetitions);
  benchmark<1, 3, false>(refinements, repetitions);
  benchmark<1, 3, true>(refinements, repetitions);
  benchmark<2, 2, false>(refinements, repetitions);
  benchmark<2, 2, true>(refinements, repetitions);
  benchmark<3, 3, false>(refinements, repetitions);
  benchmark<3, 3, true>(refinements, repetitions);
}
