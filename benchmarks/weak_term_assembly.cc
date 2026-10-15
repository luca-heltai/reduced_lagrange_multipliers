// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools_rhs.h>

#include <immersx/core/fe_space.h>
#include <immersx/core/known_term.h>
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

  template <int dim>
  class FirstCoordinateBoundaryNormal : public Function<dim>
  {
  public:
    double
    value(const Point<dim> &point, const unsigned int = 0) const override
    {
      if (std::abs(point[0] - 1.) < 1.e-12)
        return 1.;
      return 0.;
    }
  };

  struct ProductPlusGradientOffset
  {
    double
    evaluate(const double        first,
             const Tensor<1, 2> &gradient,
             const double        offset) const
    {
      return first * (1. + gradient[0]) + offset;
    }

    double
    linearize(const double        first,
              const Tensor<1, 2> &gradient,
              const double,
              const double        first_direction,
              const Tensor<1, 2> &gradient_direction,
              const double) const
    {
      return first_direction * (1. + gradient[0]) +
             first * gradient_direction[0];
    }
  };

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

template <int dim>
void
benchmark_dimension(const unsigned int refinements,
                    const unsigned int repetitions)
{
  const auto         setup_start = Clock::now();
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, -1., 1.);
  for (const auto &cell : triangulation.active_cell_iterators())
    for (const auto face : cell->face_indices())
      if (cell->face(face)->at_boundary())
        for (unsigned int d = 0; d < dim; ++d)
          if (std::abs(std::abs(cell->face(face)->center()[d]) - 1.) < 1.e-12)
            cell->face(face)->set_boundary_id(
              2 * d + (cell->face(face)->center()[d] > 0. ? 1 : 0));
  triangulation.refine_global(refinements);
  FE_Q<dim>                 finite_element(1);
  DoFHandler<dim>           dof_handler(triangulation);
  AffineConstraints<double> constraints;
  dof_handler.distribute_dofs(finite_element);
  constraints.close();

  StateLayout layout;
  const auto  space =
    fe_space(dof_handler, StaticMappingQ1<dim>::mapping, constraints);
  const auto source          = space.field(layout, "source");
  const auto target          = space.field(layout, "target");
  const auto common_setup_ms = milliseconds(Clock::now() - setup_start);
  const IntegrationSupport<dim> support(triangulation);
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;

  dealii::Functions::ConstantFunction<dim> forcing(1.);
  const auto                               rhs_model = [&] {
    auto                                model = std::make_shared<Model>();
    SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
    known_term(known_function<dim>(forcing), test(target)).add(builder);
    return model;
  }();
  const StateView<Vector>         rhs_state(layout, 0.);
  const EvaluationContext<Vector> rhs_context(0., rhs_state);
  const auto                      immersed_rhs_assembly = [&] {
    Vector result(target.dof_handler().n_dofs());
    rhs_model->evaluate_row(target.field_id(), rhs_context, result);
    return result;
  };
  const auto deal_ii_rhs_assembly = [&] {
    Vector result;
    result.reinit(dof_handler.n_dofs());
    VectorTools::create_right_hand_side(StaticMappingQ1<dim>::mapping,
                                        dof_handler,
                                        QGauss<dim>(finite_element.degree + 1),
                                        forcing,
                                        result,
                                        constraints);
    return result;
  };
  auto       immersed_rhs   = immersed_rhs_assembly();
  const auto native_rhs     = deal_ii_rhs_assembly();
  auto       rhs_difference = immersed_rhs;
  rhs_difference -= native_rhs;
  const double rhs_relative_error =
    rhs_difference.l2_norm() / native_rhs.l2_norm();
  AssertThrow(rhs_relative_error < 1.e-12,
              ExcMessage("ImmersX and VectorTools right-hand sides differ: " +
                         std::to_string(rhs_relative_error)));
  std::vector<double> immersed_rhs_samples;
  std::vector<double> native_rhs_samples;
  immersed_rhs_samples.reserve(repetitions);
  native_rhs_samples.reserve(repetitions);
  for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
    {
      if (repetition % 2 == 0)
        {
          auto start   = Clock::now();
          immersed_rhs = immersed_rhs_assembly();
          immersed_rhs_samples.push_back(milliseconds(Clock::now() - start));
          start = Clock::now();
          deal_ii_rhs_assembly();
          native_rhs_samples.push_back(milliseconds(Clock::now() - start));
        }
      else
        {
          auto start = Clock::now();
          deal_ii_rhs_assembly();
          native_rhs_samples.push_back(milliseconds(Clock::now() - start));
          start        = Clock::now();
          immersed_rhs = immersed_rhs_assembly();
          immersed_rhs_samples.push_back(milliseconds(Clock::now() - start));
        }
    }
  const auto [immersed_rhs_median, immersed_rhs_mad] =
    median_and_mad(immersed_rhs_samples);
  const auto [native_rhs_median, native_rhs_mad] =
    median_and_mad(native_rhs_samples);
  std::cout << "{\"case\":\"rhs_constant\",\"deal_ii_version\":\""
            << DEAL_II_PACKAGE_VERSION << "\",\"dimension\":" << dim
            << ",\"ranks\":1,\"dofs\":" << dof_handler.n_dofs()
            << ",\"cells\":" << triangulation.n_active_cells()
            << ",\"relative_rhs_error\":" << rhs_relative_error
            << ",\"warm_immersx_median_ms\":" << immersed_rhs_median
            << ",\"warm_immersx_mad_ms\":" << immersed_rhs_mad
            << ",\"warm_dealii_median_ms\":" << native_rhs_median
            << ",\"warm_dealii_mad_ms\":" << native_rhs_mad
            << ",\"warm_ratio\":" << immersed_rhs_median / native_rhs_median
            << ",\"repetitions\":" << repetitions << "}\n";

  if constexpr (dim > 1)
    {
      const auto boundary_id_vector = triangulation.get_boundary_ids();
      const std::set<types::boundary_id> boundary_ids(
        boundary_id_vector.begin(), boundary_id_vector.end());
      const auto boundary_rhs_model = [&] {
        auto                                model = std::make_shared<Model>();
        SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
        known_term(known_function<dim>(forcing), test(target))
          .on_boundary(boundary_ids)
          .add(builder);
        return model;
      }();
      const auto immersed_boundary_rhs_assembly = [&] {
        Vector result(target.dof_handler().n_dofs());
        boundary_rhs_model->evaluate_row(target.field_id(),
                                         rhs_context,
                                         result);
        return result;
      };
      const auto deal_ii_boundary_rhs_assembly = [&] {
        Vector result;
        result.reinit(dof_handler.n_dofs());
        VectorTools::create_boundary_right_hand_side(
          StaticMappingQ1<dim>::mapping,
          dof_handler,
          QGauss<dim - 1>(finite_element.degree + 1),
          forcing,
          result,
          boundary_ids);
        return result;
      };
      auto       immersed_boundary_rhs   = immersed_boundary_rhs_assembly();
      const auto native_boundary_rhs     = deal_ii_boundary_rhs_assembly();
      auto       boundary_rhs_difference = immersed_boundary_rhs;
      boundary_rhs_difference -= native_boundary_rhs;
      const double boundary_rhs_relative_error =
        boundary_rhs_difference.l2_norm() / native_boundary_rhs.l2_norm();
      AssertThrow(boundary_rhs_relative_error < 1.e-12,
                  ExcMessage(
                    "ImmersX and VectorTools boundary right-hand sides "
                    "differ: " +
                    std::to_string(boundary_rhs_relative_error)));
      std::vector<double> immersed_boundary_rhs_samples;
      std::vector<double> native_boundary_rhs_samples;
      immersed_boundary_rhs_samples.reserve(repetitions);
      native_boundary_rhs_samples.reserve(repetitions);
      for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
        {
          if (repetition % 2 == 0)
            {
              auto start            = Clock::now();
              immersed_boundary_rhs = immersed_boundary_rhs_assembly();
              immersed_boundary_rhs_samples.push_back(
                milliseconds(Clock::now() - start));
              start = Clock::now();
              deal_ii_boundary_rhs_assembly();
              native_boundary_rhs_samples.push_back(
                milliseconds(Clock::now() - start));
            }
          else
            {
              auto start = Clock::now();
              deal_ii_boundary_rhs_assembly();
              native_boundary_rhs_samples.push_back(
                milliseconds(Clock::now() - start));
              start                 = Clock::now();
              immersed_boundary_rhs = immersed_boundary_rhs_assembly();
              immersed_boundary_rhs_samples.push_back(
                milliseconds(Clock::now() - start));
            }
        }
      const auto [immersed_boundary_rhs_median, immersed_boundary_rhs_mad] =
        median_and_mad(immersed_boundary_rhs_samples);
      const auto [native_boundary_rhs_median, native_boundary_rhs_mad] =
        median_and_mad(native_boundary_rhs_samples);
      std::cout << "{\"case\":\"rhs_boundary\",\"deal_ii_version\":\""
                << DEAL_II_PACKAGE_VERSION << "\",\"dimension\":" << dim
                << ",\"ranks\":1,\"dofs\":" << dof_handler.n_dofs()
                << ",\"cells\":" << triangulation.n_active_cells()
                << ",\"relative_rhs_error\":" << boundary_rhs_relative_error
                << ",\"warm_immersx_median_ms\":"
                << immersed_boundary_rhs_median
                << ",\"warm_immersx_mad_ms\":" << immersed_boundary_rhs_mad
                << ",\"warm_dealii_median_ms\":" << native_boundary_rhs_median
                << ",\"warm_dealii_mad_ms\":" << native_boundary_rhs_mad
                << ",\"warm_ratio\":"
                << immersed_boundary_rhs_median / native_boundary_rhs_median
                << ",\"repetitions\":" << repetitions << "}\n";

      std::set<types::boundary_id> normal_boundary_ids;
      for (const auto &cell : triangulation.active_cell_iterators())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary() &&
              std::abs(cell->face(face)->center()[0] - 1.) < 1.e-12)
            normal_boundary_ids.insert(cell->face(face)->boundary_id());
      AssertThrow(normal_boundary_ids.size() == 1,
                  ExcMessage("Expected one boundary id at x=1."));
      const BoundaryIntegrationSupport<dim> boundary_support(
        triangulation, *normal_boundary_ids.begin());
      std::vector<Point<dim>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(StaticMappingQ1<dim>::mapping,
                                           dof_handler,
                                           support_points);
      Vector linear_state(dof_handler.n_dofs());
      for (unsigned int i = 0; i < linear_state.size(); ++i)
        linear_state[i] = support_points[i][0];
      const auto immersed_boundary_flux_assembly = [&] {
        auto                                model = std::make_shared<Model>();
        SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
        (gradient(source) * boundary_support.normal() * test(target) *
         boundary_support)
          .add(builder);
        StateView<Vector> state_view(layout, 0.);
        state_view.bind(source.field_id(), linear_state);
        const EvaluationContext<Vector> context(0., state_view);
        Vector                          result(target.dof_handler().n_dofs());
        model->evaluate_row(target.field_id(), context, result);
        return result;
      };
      const auto native_boundary_flux_assembly = [&] {
        Vector result;
        result.reinit(dof_handler.n_dofs());
        FirstCoordinateBoundaryNormal<dim> forcing;
        VectorTools::create_boundary_right_hand_side(
          StaticMappingQ1<dim>::mapping,
          dof_handler,
          QGauss<dim - 1>(finite_element.degree + 1),
          forcing,
          result,
          normal_boundary_ids);
        return result;
      };
      auto       immersed_boundary_flux   = immersed_boundary_flux_assembly();
      const auto native_boundary_flux     = native_boundary_flux_assembly();
      auto       boundary_flux_difference = immersed_boundary_flux;
      boundary_flux_difference -= native_boundary_flux;
      const double boundary_flux_relative_error =
        boundary_flux_difference.l2_norm() / native_boundary_flux.l2_norm();
      AssertThrow(boundary_flux_relative_error < 1.e-12,
                  ExcMessage(
                    "ImmersX and VectorTools boundary fluxes differ: " +
                    std::to_string(boundary_flux_relative_error)));
      std::vector<double> immersed_boundary_flux_samples;
      std::vector<double> native_boundary_flux_samples;
      for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
        {
          if (repetition % 2 == 0)
            {
              auto start             = Clock::now();
              immersed_boundary_flux = immersed_boundary_flux_assembly();
              immersed_boundary_flux_samples.push_back(
                milliseconds(Clock::now() - start));
              start = Clock::now();
              native_boundary_flux_assembly();
              native_boundary_flux_samples.push_back(
                milliseconds(Clock::now() - start));
            }
          else
            {
              auto start = Clock::now();
              native_boundary_flux_assembly();
              native_boundary_flux_samples.push_back(
                milliseconds(Clock::now() - start));
              start                  = Clock::now();
              immersed_boundary_flux = immersed_boundary_flux_assembly();
              immersed_boundary_flux_samples.push_back(
                milliseconds(Clock::now() - start));
            }
        }
      const auto [immersed_boundary_flux_median, immersed_boundary_flux_mad] =
        median_and_mad(immersed_boundary_flux_samples);
      const auto [native_boundary_flux_median, native_boundary_flux_mad] =
        median_and_mad(native_boundary_flux_samples);
      std::cout << "{\"case\":\"boundary_flux_q1\",\"deal_ii_version\":\""
                << DEAL_II_PACKAGE_VERSION << "\",\"dimension\":" << dim
                << ",\"ranks\":1,\"dofs\":" << dof_handler.n_dofs()
                << ",\"cells\":" << triangulation.n_active_cells()
                << ",\"relative_rhs_error\":" << boundary_flux_relative_error
                << ",\"warm_immersx_median_ms\":"
                << immersed_boundary_flux_median
                << ",\"warm_immersx_mad_ms\":" << immersed_boundary_flux_mad
                << ",\"warm_dealii_median_ms\":" << native_boundary_flux_median
                << ",\"warm_dealii_mad_ms\":" << native_boundary_flux_mad
                << ",\"warm_ratio\":"
                << immersed_boundary_flux_median / native_boundary_flux_median
                << ",\"repetitions\":" << repetitions << "}\n";
    }

  const auto benchmark_pairing = [&](const std::string &name,
                                     const auto        &trial,
                                     const auto        &test_expression,
                                     const auto        &native_creator) {
    const auto immersed_assembly = [&] {
      auto                                model = std::make_shared<Model>();
      SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
      weak_term(trial, test_expression, support).add(builder);

      Vector source_state(source.dof_handler().n_dofs());
      source_state = 1.;
      StateView<Vector> state_view(layout, 0.);
      state_view.bind(source.field_id(), source_state);
      const EvaluationContext<Vector> context(0., state_view);
      const auto matrix = model->state_matrix_operator(target.field_id(),
                                                       source.field_id(),
                                                       context);
      AssertThrow(matrix.has_value(),
                  ExcMessage("The ImmersX weak term did not expose a matrix."));
      return std::make_pair(std::move(model), matrix->matrix());
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
      native_creator(*matrix);
      return std::make_pair(std::move(matrix), std::move(sparsity));
    };

    const auto cold_immersx_start = Clock::now();
    auto       immersed           = immersed_assembly();
    const auto cold_immersx_ms =
      milliseconds(Clock::now() - cold_immersx_start);

    const auto cold_dealii_start = Clock::now();
    auto       native            = deal_ii_assembly();
    const auto cold_dealii_ms = milliseconds(Clock::now() - cold_dealii_start);

    const double relative_error =
      relative_matrix_difference(*immersed.second, *native.first);
    AssertThrow(relative_error < 1.e-12,
                ExcMessage("ImmersX and deal.II matrices differ: " +
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

    const auto [immersed_median, immersed_mad] =
      median_and_mad(immersed_samples);
    const auto [native_median, native_mad] = median_and_mad(native_samples);
    std::cout << "{\"case\":\"" << name << "\",\"deal_ii_version\":\""
              << DEAL_II_PACKAGE_VERSION << "\",\"dimension\":" << dim
              << ",\"ranks\":1,\"dofs\":" << dof_handler.n_dofs()
              << ",\"cells\":" << triangulation.n_active_cells()
              << ",\"quadrature_points_per_cell\":"
              << QGauss<dim>(finite_element.degree + 1).size()
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
  };

  benchmark_pairing("mass_q1",
                    value(source),
                    test(target),
                    [&](Matrix &matrix) {
                      MatrixCreator::create_mass_matrix(
                        StaticMappingQ1<dim>::mapping,
                        dof_handler,
                        QGauss<dim>(finite_element.degree + 1),
                        matrix,
                        static_cast<const Function<dim, double> *>(nullptr),
                        constraints);
                    });
  benchmark_pairing("laplace_q1",
                    gradient(source),
                    gradient(test(target)),
                    [&](Matrix &matrix) {
                      MatrixCreator::create_laplace_matrix(
                        StaticMappingQ1<dim>::mapping,
                        dof_handler,
                        QGauss<dim>(finite_element.degree + 1),
                        matrix,
                        static_cast<const Function<dim, double> *>(nullptr),
                        constraints);
                    });
}

void
benchmark_heterogeneous_nonlinear(const unsigned int refinements,
                                  const unsigned int repetitions)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;

  const auto       setup_start = Clock::now();
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(refinements);
  for (const auto &cell : triangulation.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 0);

  FE_Q<2>       first_fe(1);
  FE_Q<2>       second_fe(2);
  FE_DGQ<2>     frozen_fe(0);
  DoFHandler<2> first_dh(triangulation), second_dh(triangulation),
    frozen_dh(triangulation);
  first_dh.distribute_dofs(first_fe);
  second_dh.distribute_dofs(second_fe);
  frozen_dh.distribute_dofs(frozen_fe);
  AffineConstraints<double> first_constraints, second_constraints,
    frozen_constraints;
  first_constraints.close();
  second_constraints.close();
  frozen_constraints.close();

  StateLayout layout;
  const auto  first_space =
    fe_space(first_dh, StaticMappingQ1<2>::mapping, first_constraints);
  const auto second_space =
    fe_space(second_dh, StaticMappingQ1<2>::mapping, second_constraints);
  const auto frozen_space =
    fe_space(frozen_dh, StaticMappingQ1<2>::mapping, frozen_constraints);
  const auto first  = first_space.field(layout, "first");
  const auto target = first_space.field(layout, "target");
  const auto second = second_space.field(layout, "second");
  const auto offset = frozen_space.field("offset");
  Vector     frozen_coefficients(frozen_dh.n_dofs());
  for (unsigned int i = 0; i < frozen_coefficients.size(); ++i)
    frozen_coefficients[i] = 0.25 + 0.1 * i;

  const auto                          observable = transform(value(first),
                                    gradient(second),
                                    frozen(offset, frozen_coefficients),
                                    ProductPlusGradientOffset{});
  const IntegrationSupport<2>         support(triangulation, 1);
  auto                                model = std::make_shared<Model>();
  SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
  weak_term(observable, test(target), support).add(builder);

  Vector first_state(first_dh.n_dofs()), second_state(second_dh.n_dofs()),
    target_state(first_dh.n_dofs());
  for (unsigned int i = 0; i < first_state.size(); ++i)
    first_state[i] = 0.5 + 0.2 * i;
  for (unsigned int i = 0; i < second_state.size(); ++i)
    second_state[i] = 1. + 0.05 * i;
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(first.field_id(), first_state);
  state_view.bind(second.field_id(), second_state);
  state_view.bind(target.field_id(), target_state);
  const EvaluationContext<Vector> context(0., state_view);
  const QGauss<2>                 quadrature(3);

  struct Scratch
  {
    Scratch(const Mapping<2>       &mapping,
            const FiniteElement<2> &first_fe,
            const FiniteElement<2> &second_fe,
            const FiniteElement<2> &frozen_fe,
            const FiniteElement<2> &target_fe,
            const Quadrature<2>    &quadrature)
      : first(mapping, first_fe, quadrature, update_values | update_JxW_values)
      , second(mapping, second_fe, quadrature, update_gradients)
      , frozen(mapping, frozen_fe, quadrature, update_values)
      , target(mapping,
               target_fe,
               quadrature,
               update_values | update_JxW_values)
    {}

    MeshWorker::ScratchData<2> first, second, frozen, target;
  };
  Scratch      scratch(StaticMappingQ1<2>::mapping,
                  first_fe,
                  second_fe,
                  frozen_fe,
                  first_fe,
                  quadrature);
  const double setup_ms = milliseconds(Clock::now() - setup_start);

  struct NativeAssembly
  {
    std::shared_ptr<SparsityPattern> first_sparsity, second_sparsity;
    std::shared_ptr<Matrix>          first_jacobian, second_jacobian;
    Vector                           residual;
  };
  const auto native_assembly = [&] {
    DynamicSparsityPattern first_dynamic(target.dof_handler().n_dofs(),
                                         first.dof_handler().n_dofs());
    DynamicSparsityPattern second_dynamic(target.dof_handler().n_dofs(),
                                          second.dof_handler().n_dofs());
    DoFTools::make_sparsity_pattern(target.dof_handler(),
                                    first.dof_handler(),
                                    first_dynamic);
    DoFTools::make_sparsity_pattern(target.dof_handler(),
                                    second.dof_handler(),
                                    second_dynamic);
    auto first_pattern  = std::make_shared<SparsityPattern>();
    auto second_pattern = std::make_shared<SparsityPattern>();
    first_pattern->copy_from(first_dynamic);
    second_pattern->copy_from(second_dynamic);
    auto   first_matrix  = std::make_shared<Matrix>(*first_pattern);
    auto   second_matrix = std::make_shared<Matrix>(*second_pattern);
    Vector residual(target.dof_handler().n_dofs());

    MeshWorker::CopyData<2, 1, 3> copy;
    const auto                    cell_worker = [&](const auto &cell,
                                 auto       &worker_scratch,
                                 auto       &data) {
      if (cell->material_id() != 1)
        return;
      const auto &first_values = worker_scratch.first.reinit(cell);
      const auto  second_cell =
        cell->as_dof_handler_iterator(second.dof_handler());
      const auto frozen_cell =
        cell->as_dof_handler_iterator(offset.dof_handler());
      const auto target_cell =
        cell->as_dof_handler_iterator(target.dof_handler());
      const auto &second_values = worker_scratch.second.reinit(second_cell);
      const auto &frozen_fe_values = worker_scratch.frozen.reinit(frozen_cell);
      const auto &target_values = worker_scratch.target.reinit(target_cell);
      data.reinit(0, first_fe.n_dofs_per_cell(), first_fe.n_dofs_per_cell());
      data.reinit(1, first_fe.n_dofs_per_cell(), second_fe.n_dofs_per_cell());
      data.vectors[0].reinit(first_fe.n_dofs_per_cell());
      data.local_dof_indices[2].resize(first_fe.n_dofs_per_cell());
      auto &first_indices  = data.local_dof_indices[0];
      auto &second_indices = data.local_dof_indices[1];
      auto &target_indices = data.local_dof_indices[2];
      std::vector<types::global_dof_index> frozen_indices(
        frozen_fe.n_dofs_per_cell());
      cell->get_dof_indices(first_indices);
      second_cell->get_dof_indices(second_indices);
      frozen_cell->get_dof_indices(frozen_indices);
      target_cell->get_dof_indices(target_indices);

      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          double       first_value = 0.;
          Tensor<1, 2> second_gradient;
          for (unsigned int j = 0; j < first_indices.size(); ++j)
            first_value +=
              first_state[first_indices[j]] * first_values.shape_value(j, q);
          for (unsigned int j = 0; j < second_indices.size(); ++j)
            second_gradient +=
              second_state[second_indices[j]] * second_values.shape_grad(j, q);
          const double integrand = first_value * (1. + second_gradient[0]) +
                                   frozen_coefficients[frozen_indices[0]] *
                                     frozen_fe_values.shape_value(0, q);
          for (unsigned int i = 0; i < target_indices.size(); ++i)
            {
              const double test_value = target_values.shape_value(i, q);
              data.vectors[0][i] +=
                integrand * test_value * target_values.JxW(q);
              for (unsigned int j = 0; j < first_indices.size(); ++j)
                data.matrices[0](i, j) += (1. + second_gradient[0]) *
                                          first_values.shape_value(j, q) *
                                          test_value * target_values.JxW(q);
              for (unsigned int j = 0; j < second_indices.size(); ++j)
                data.matrices[1](i, j) += first_value *
                                          second_values.shape_grad(j, q)[0] *
                                          test_value * target_values.JxW(q);
            }
        }
    };
    const auto copier = [&](const auto &data) {
      first_constraints.distribute_local_to_global(data.vectors[0],
                                                   data.local_dof_indices[2],
                                                   residual);
      first_constraints.distribute_local_to_global(data.matrices[0],
                                                   data.local_dof_indices[2],
                                                   first_constraints,
                                                   data.local_dof_indices[0],
                                                   *first_matrix);
      first_constraints.distribute_local_to_global(data.matrices[1],
                                                   data.local_dof_indices[2],
                                                   second_constraints,
                                                   data.local_dof_indices[1],
                                                   *second_matrix);
    };
    MeshWorker::mesh_loop(first.dof_handler().active_cell_iterators(),
                          cell_worker,
                          copier,
                          scratch,
                          copy,
                          MeshWorker::assemble_own_cells,
                          {},
                          {},
                          1,
                          1);
    return NativeAssembly{std::move(first_pattern),
                          std::move(second_pattern),
                          std::move(first_matrix),
                          std::move(second_matrix),
                          std::move(residual)};
  };
  Vector     immersed_residual(target.dof_handler().n_dofs());
  const auto immersed_assembly = [&] {
    immersed_residual = 0.;
    model->evaluate_row(target.field_id(), context, immersed_residual);
    auto first_jacobian  = model->state_matrix_operator(target.field_id(),
                                                       first.field_id(),
                                                       context);
    auto second_jacobian = model->state_matrix_operator(target.field_id(),
                                                        second.field_id(),
                                                        context);
    AssertThrow(first_jacobian.has_value() && second_jacobian.has_value(),
                ExcMessage("Missing heterogeneous Jacobian."));
    return std::pair{std::move(first_jacobian), std::move(second_jacobian)};
  };
  const auto immersed_result     = immersed_assembly();
  const auto native_result       = native_assembly();
  Vector     residual_difference = immersed_residual;
  residual_difference -= native_result.residual;
  const double residual_error =
    residual_difference.l2_norm() /
    std::max(native_result.residual.l2_norm(), 1.e-300);
  const double first_jacobian_error =
    relative_matrix_difference(*immersed_result.first->matrix(),
                               *native_result.first_jacobian);
  const double second_jacobian_error =
    relative_matrix_difference(*immersed_result.second->matrix(),
                               *native_result.second_jacobian);
  Vector direction_first(first_state.size()),
    direction_second(second_state.size());
  for (unsigned int i = 0; i < direction_first.size(); ++i)
    direction_first[i] = 0.3 + 0.07 * i;
  for (unsigned int i = 0; i < direction_second.size(); ++i)
    direction_second[i] = 0.8 - 0.025 * i;
  Vector immersed_first_action(target.dof_handler().n_dofs()),
    immersed_second_action(target.dof_handler().n_dofs()),
    native_first_action(target.dof_handler().n_dofs()),
    native_second_action(target.dof_handler().n_dofs());
  immersed_result.first->view.vmult(immersed_first_action, direction_first);
  immersed_result.second->view.vmult(immersed_second_action, direction_second);
  native_result.first_jacobian->vmult(native_first_action, direction_first);
  native_result.second_jacobian->vmult(native_second_action, direction_second);
  immersed_first_action -= native_first_action;
  immersed_second_action -= native_second_action;
  const double jacobian_action_error =
    std::sqrt(immersed_first_action.norm_sqr() +
              immersed_second_action.norm_sqr()) /
    std::max(std::sqrt(native_first_action.norm_sqr() +
                       native_second_action.norm_sqr()),
             1.e-300);
  AssertThrow(residual_error < 1.e-12 && first_jacobian_error < 1.e-12 &&
                second_jacobian_error < 1.e-12 &&
                jacobian_action_error < 1.e-12,
              ExcMessage(
                "ImmersX and MeshWorker heterogeneous results differ."));

  // Both paths return the residual and both Jacobians. MeshWorker fuses them
  // into one cell traversal; ImmersX exposes separate residual and Jacobian
  // assembly calls, so the ratio measures each public path's end-to-end cost.
  immersed_assembly();
  native_assembly();
  std::vector<double> immersed_samples, native_samples;
  immersed_samples.reserve(repetitions);
  native_samples.reserve(repetitions);
  for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
    if (repetition % 2 == 0)
      {
        auto start = Clock::now();
        (void)immersed_assembly();
        immersed_samples.push_back(milliseconds(Clock::now() - start));
        start = Clock::now();
        (void)native_assembly();
        native_samples.push_back(milliseconds(Clock::now() - start));
      }
    else
      {
        auto start = Clock::now();
        (void)native_assembly();
        native_samples.push_back(milliseconds(Clock::now() - start));
        start = Clock::now();
        (void)immersed_assembly();
        immersed_samples.push_back(milliseconds(Clock::now() - start));
      }
  const auto [immersed_median, immersed_mad] = median_and_mad(immersed_samples);
  const auto [native_median, native_mad]     = median_and_mad(native_samples);
  std::cout << "{\"case\":\"heterogeneous_nonlinear_q1_q2_dgq0\","
            << "\"deal_ii_version\":\"" << DEAL_II_PACKAGE_VERSION
            << "\",\"dimension\":2,\"ranks\":1,\"dofs_q1\":"
            << first_dh.n_dofs() << ",\"dofs_q2\":" << second_dh.n_dofs()
            << ",\"cells\":" << triangulation.n_active_cells()
            << ",\"quadrature_points_per_cell\":" << quadrature.size()
            << ",\"setup_ms\":" << setup_ms
            << ",\"relative_residual_error\":" << residual_error
            << ",\"relative_q1_jacobian_error\":" << first_jacobian_error
            << ",\"relative_q2_jacobian_error\":" << second_jacobian_error
            << ",\"relative_jacobian_action_error\":" << jacobian_action_error
            << ",\"warm_immersx_median_ms\":" << immersed_median
            << ",\"warm_immersx_mad_ms\":" << immersed_mad
            << ",\"warm_dealii_meshworker_median_ms\":" << native_median
            << ",\"warm_dealii_meshworker_mad_ms\":" << native_mad
            << ",\"warm_ratio\":" << immersed_median / native_median
            << ",\"repetitions\":" << repetitions << "}\n";
}

void
benchmark_nonmatching_support(const unsigned int repetitions)
{
  using Vector = dealii::Vector<double>;
  using Matrix = dealii::SparseMatrix<double>;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  struct ImmersXAssembly
  {
    std::shared_ptr<Model>  model;
    std::shared_ptr<Matrix> matrix;
  };
  struct NativeAssembly
  {
    std::shared_ptr<SparsityPattern> pattern;
    std::shared_ptr<Matrix>          matrix;
  };

  Triangulation<2> bulk_triangulation;
  GridGenerator::hyper_cube(bulk_triangulation, 0., 1.);
  bulk_triangulation.refine_global(3);
  Triangulation<1, 2> embedded_triangulation;
  GridGenerator::hyper_cube(embedded_triangulation, 0., 1.);
  GridTools::transform(
    [](const Point<2> &point) {
      auto shifted = point;
      shifted[1] += 0.5;
      return shifted;
    },
    embedded_triangulation);
  embedded_triangulation.refine_global(3);

  FE_Q<2>          bulk_fe(1);
  FE_Q<1, 2>       embedded_fe(1);
  DoFHandler<2>    bulk_dof_handler(bulk_triangulation);
  DoFHandler<1, 2> embedded_dof_handler(embedded_triangulation);
  bulk_dof_handler.distribute_dofs(bulk_fe);
  embedded_dof_handler.distribute_dofs(embedded_fe);
  AffineConstraints<double> bulk_constraints;
  AffineConstraints<double> embedded_constraints;
  bulk_constraints.close();
  embedded_constraints.close();

  StateLayout layout;
  const auto  bulk_view =
    fe_space(bulk_dof_handler, StaticMappingQ1<2>::mapping, bulk_constraints);
  const auto embedded_view  = fe_space(embedded_dof_handler,
                                      StaticMappingQ1<1, 2>::mapping,
                                      embedded_constraints);
  const auto bulk_field     = bulk_view.field(layout, "bulk");
  const auto embedded_field = embedded_view.field(layout, "embedded");
  const IntegrationSupport<1, 2> support(embedded_triangulation);
  const QGauss<1>                quadrature(2);

  const auto native_assembly = [&] {
    DynamicSparsityPattern sparsity(bulk_dof_handler.n_dofs(),
                                    embedded_dof_handler.n_dofs());
    GridTools::Cache<2, 2> cache(bulk_triangulation,
                                 StaticMappingQ1<2>::mapping);
    NonMatching::create_coupling_sparsity_pattern(
      cache,
      bulk_dof_handler,
      embedded_dof_handler,
      quadrature,
      sparsity,
      bulk_constraints,
      ComponentMask(),
      ComponentMask(),
      StaticMappingQ1<1, 2>::mapping,
      embedded_constraints);
    auto pattern = std::make_shared<SparsityPattern>();
    pattern->copy_from(sparsity);
    auto matrix = std::make_shared<Matrix>(*pattern);
    NonMatching::create_coupling_mass_matrix(cache,
                                             bulk_dof_handler,
                                             embedded_dof_handler,
                                             quadrature,
                                             *matrix,
                                             bulk_constraints,
                                             ComponentMask(),
                                             ComponentMask(),
                                             StaticMappingQ1<1, 2>::mapping,
                                             embedded_constraints);
    return NativeAssembly{std::move(pattern), std::move(matrix)};
  };

  const auto immersed_reverse_assembly = [&] {
    auto                                model = std::make_shared<Model>();
    SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
    weak_term(value(bulk_field), test(embedded_field), support).add(builder);
    const StateView<Vector>         state(layout, 0.);
    const EvaluationContext<Vector> context(0., state);
    const auto matrix = model->state_matrix_operator(embedded_field.field_id(),
                                                     bulk_field.field_id(),
                                                     context);
    return ImmersXAssembly{std::move(model), matrix->matrix()};
  };
  const auto immersed_forward_assembly = [&] {
    auto                                model = std::make_shared<Model>();
    SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
    weak_term(value(embedded_field), test(bulk_field), support).add(builder);
    const StateView<Vector>         state(layout, 0.);
    const EvaluationContext<Vector> context(0., state);
    const auto matrix = model->state_matrix_operator(bulk_field.field_id(),
                                                     embedded_field.field_id(),
                                                     context);
    return ImmersXAssembly{std::move(model), matrix->matrix()};
  };

  const auto benchmark_pair = [&](const std::string &name,
                                  const auto        &immersed_assembly,
                                  const bool         transpose) {
    auto         start                    = Clock::now();
    auto         immersed_assembly_result = immersed_assembly();
    const auto  &immersed_matrix          = *immersed_assembly_result.matrix;
    const double cold_imm                 = milliseconds(Clock::now() - start);
    start                                 = Clock::now();
    auto         native_matrix            = native_assembly();
    const double cold_native              = milliseconds(Clock::now() - start);
    auto         difference = [&](const auto &immersed, const auto &native) {
      AssertDimension(immersed.m(), transpose ? native.n() : native.m());
      AssertDimension(immersed.n(), transpose ? native.m() : native.n());
      double difference_squared = 0.;
      double native_squared     = 0.;
      for (unsigned int i = 0; i < immersed.m(); ++i)
        for (unsigned int j = 0; j < immersed.n(); ++j)
          {
            const double expected =
              transpose ? native.el(j, i) : native.el(i, j);
            const double delta = immersed.el(i, j) - expected;
            difference_squared += delta * delta;
            native_squared += expected * expected;
          }
      return std::sqrt(difference_squared / native_squared);
    };
    const double relative_error =
      difference(immersed_matrix, *native_matrix.matrix);
    AssertThrow(relative_error < 1.e-12,
                ExcMessage("ImmersX and NonMatching matrices differ: " +
                           std::to_string(relative_error)));

    std::vector<double> immersed_samples;
    std::vector<double> native_samples;
    immersed_samples.reserve(repetitions);
    native_samples.reserve(repetitions);
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
      if (repetition % 2 == 0)
        {
          start = Clock::now();
          (void)immersed_assembly();
          immersed_samples.push_back(milliseconds(Clock::now() - start));
          start = Clock::now();
          native_assembly();
          native_samples.push_back(milliseconds(Clock::now() - start));
        }
      else
        {
          start = Clock::now();
          native_assembly();
          native_samples.push_back(milliseconds(Clock::now() - start));
          start = Clock::now();
          (void)immersed_assembly();
          immersed_samples.push_back(milliseconds(Clock::now() - start));
        }
    const auto [immersed_median, immersed_mad] =
      median_and_mad(immersed_samples);
    const auto [native_median, native_mad] = median_and_mad(native_samples);
    std::cout << "{\"case\":\"" << name << "\",\"deal_ii_version\":\""
              << DEAL_II_PACKAGE_VERSION << "\",\"dimension\":2"
              << ",\"embedded_dimension\":1,\"ranks\":1,\"bulk_dofs\":"
              << bulk_dof_handler.n_dofs()
              << ",\"embedded_dofs\":" << embedded_dof_handler.n_dofs()
              << ",\"relative_matrix_error\":" << relative_error
              << ",\"cold_immersx_ms\":" << cold_imm
              << ",\"cold_dealii_ms\":" << cold_native
              << ",\"cold_ratio\":" << cold_imm / cold_native
              << ",\"warm_immersx_median_ms\":" << immersed_median
              << ",\"warm_immersx_mad_ms\":" << immersed_mad
              << ",\"warm_dealii_median_ms\":" << native_median
              << ",\"warm_dealii_mad_ms\":" << native_mad
              << ",\"warm_ratio\":" << immersed_median / native_median
              << ",\"repetitions\":" << repetitions << "}\n";
  };

  benchmark_pair("nonmatching_value_reverse", immersed_reverse_assembly, true);
  benchmark_pair("nonmatching_value_forward", immersed_forward_assembly, false);
}

int
main(int argc, char **argv)
{
  const auto refinements = argument(argc, argv, "refinements", 4);
  const auto repetitions = argument(argc, argv, "repetitions", 9);
  AssertThrow(repetitions > 0,
              ExcMessage("The benchmark needs at least one repetition."));

  benchmark_dimension<1>(refinements, repetitions);
  benchmark_dimension<2>(refinements, repetitions);
  benchmark_dimension<3>(refinements, repetitions);
  benchmark_heterogeneous_nonlinear(refinements, repetitions);
  benchmark_nonmatching_support(repetitions);
}
