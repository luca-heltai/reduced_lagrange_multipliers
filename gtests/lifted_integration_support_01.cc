#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/lifted_integration_support.h>
#include <immersx/core/state.h>
#include <immersx/core/weak_term.h>

#include <algorithm>
#include <set>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct LiftedLineFixture
  {
    LiftedLineFixture()
      : domain_parameters("/Lifted line domain/")
      , lift("/Lifted line geometry/")
      , domain(domain_parameters)
    {
      domain_parameters.name_of_grid       = "hyper_cube";
      domain_parameters.arguments_for_grid = "-1: 1: false";
      domain_parameters.initial_refinement = 0;
      domain.make_grid();

      lift.section.inclusion_degree      = 0;
      lift.section.refinement_level      = 0;
      lift.section.selected_coefficients = {0};
      lift.section.n_q_points            = 2;
      lift.representative_n_q_points     = 2;
      lift.thickness                     = "0.25";
    }

    DomainParameters<1, 3>     domain_parameters;
    TensorProductLift<1, 2, 3> lift;
    Domain<1, 3>               domain;
  };
} // namespace

TEST(LiftedIntegrationSupport, BOTH_UsesPhysicalQuadratureWithoutAField)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  const auto        support = fixture.domain.lift(fixture.lift);
  static_assert(
    is_physical_quadrature_support<decltype(support)>::value,
    "Lifted support must satisfy the physical point-range contract.");

  double                  local_measure = 0.;
  std::set<std::uint64_t> local_ids;
  for (std::size_t q = 0; q < support.n_locally_owned_quadrature_points(); ++q)
    {
      const auto &point = support.locally_owned_quadrature_points()[q];
      local_measure += point.weight;
      EXPECT_TRUE(local_ids.insert(point.stable_id).second);
      EXPECT_EQ(point.owner_rank,
                Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
      EXPECT_NEAR(support.normal(q).norm(), 1., 1.e-12);
    }
  const std::vector<std::uint64_t> original_ids(local_ids.begin(),
                                                local_ids.end());

  const auto global_measure =
    Utilities::MPI::sum(local_measure, fixture.domain.communicator());
  EXPECT_NEAR(global_measure, 1., 1.e-10);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  support.n_locally_owned_quadrature_points()),
                                fixture.domain.communicator()),
            4u);

  // The descriptor is non-owning and changes take effect after rebuild().
  fixture.lift.thickness = "0.5";
  support.rebuild();
  local_measure = 0.;
  std::vector<std::uint64_t> rebuilt_ids;
  for (const auto &point : support.locally_owned_quadrature_points())
    {
      local_measure += point.weight;
      rebuilt_ids.push_back(point.stable_id);
    }
  std::sort(rebuilt_ids.begin(), rebuilt_ids.end());
  EXPECT_EQ(rebuilt_ids, original_ids);
  EXPECT_NEAR(Utilities::MPI::sum(local_measure, fixture.domain.communicator()),
              2.,
              1.e-10);
}

TEST(LiftedIntegrationSupport, RejectsInvalidThickness)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  fixture.lift.thickness = "0";
  EXPECT_THROW(fixture.domain.lift(fixture.lift), ExceptionBase);
}

TEST(LiftedIntegrationSupport, BOTH_OwnershipAndIdsArePartitionIndependent)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> domain_parameters("/Distributed lifted line domain/");
  domain_parameters.name_of_grid       = "hyper_cube";
  domain_parameters.arguments_for_grid = "-1: 1: false";
  domain_parameters.initial_refinement = 2;
  Domain<1, 3> domain(domain_parameters);
  domain.make_grid();

  TensorProductLift<1, 2, 3> lift("/Distributed lifted line geometry/");
  lift.section.inclusion_degree      = 0;
  lift.section.refinement_level      = 0;
  lift.section.selected_coefficients = {0};
  lift.section.n_q_points            = 2;
  lift.representative_n_q_points     = 2;
  lift.thickness                     = "0.25";
  const auto support                 = domain.lift(lift);

  const auto rank = Utilities::MPI::this_mpi_process(domain.communicator());
  std::vector<std::uint64_t> local_ids;
  for (const auto &point : support.locally_owned_quadrature_points())
    {
      EXPECT_EQ(point.owner_rank, rank);
      local_ids.push_back(point.stable_id);
    }
  const auto gathered_ids =
    Utilities::MPI::all_gather(domain.communicator(), local_ids);
  std::vector<std::uint64_t> all_ids;
  for (const auto &rank_ids : gathered_ids)
    all_ids.insert(all_ids.end(), rank_ids.begin(), rank_ids.end());
  std::sort(all_ids.begin(), all_ids.end());
  ASSERT_EQ(all_ids.size(), 16u);
  EXPECT_TRUE(std::adjacent_find(all_ids.begin(), all_ids.end()) ==
              all_ids.end());
}

TEST(LiftedIntegrationSupport, IntegratesFrozenNonconstantThickness)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  const auto        support =
    fixture.domain.lift(fixture.lift, [](const Point<3> &point) {
      return 0.25 + 0.1 * point[0];
    });

  double local_measure = 0.;
  for (const auto &point : support.locally_owned_quadrature_points())
    local_measure += point.weight;
  EXPECT_NEAR(Utilities::MPI::sum(local_measure, fixture.domain.communicator()),
              1.,
              1.e-10);
}

TEST(LiftedIntegrationSupport, RotatedRepresentativeLineUsesPhysicalGeometry)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> domain_parameters("/Rotated lifted line domain/");
  domain_parameters.name_of_grid =
    TestPaths::data_filename("tests/one_cylinder_rotation_z.vtk");
  domain_parameters.initial_refinement = 0;
  Domain<1, 3> domain(domain_parameters);
  domain.make_grid();

  TensorProductLift<1, 2, 3> lift("/Rotated lifted line geometry/");
  lift.section.inclusion_degree      = 0;
  lift.section.refinement_level      = 0;
  lift.section.selected_coefficients = {0};
  lift.section.n_q_points            = 2;
  lift.representative_n_q_points     = 2;
  lift.thickness                     = "0.25";
  const auto support                 = domain.lift(lift);

  double      local_measure = 0.;
  const auto &points        = support.locally_owned_quadrature_points();
  for (std::size_t q = 0; q < points.size(); ++q)
    {
      const auto &point = points[q];
      local_measure += point.weight;
      EXPECT_NEAR(point.representative_point[0], 0., 1.e-12);
      EXPECT_NEAR(point.representative_point[1], 0., 1.e-12);
      EXPECT_NEAR(support.normal(q).norm(), 1., 1.e-12);
    }
  EXPECT_NEAR(Utilities::MPI::sum(local_measure, domain.communicator()),
              2.,
              1.e-10);
}

TEST(LiftedIntegrationSupport, BOTH_AssemblesBulkMassOnPhysicalQuadrature)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  const auto        support = fixture.domain.lift(fixture.lift);

  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(1);
  FE_Q<3>       source_element(1);
  FE_Q<3>       target_element(2);
  DoFHandler<3> source_dof_handler(background);
  DoFHandler<3> target_dof_handler(background);
  source_dof_handler.distribute_dofs(source_element);
  target_dof_handler.distribute_dofs(target_element);
  AffineConstraints<double> source_constraints;
  DoFTools::make_hanging_node_constraints(source_dof_handler,
                                          source_constraints);
  source_constraints.close();
  AffineConstraints<double> target_constraints;
  DoFTools::make_hanging_node_constraints(target_dof_handler,
                                          target_constraints);
  target_constraints.close();
  const auto source_space = fe_space(source_dof_handler,
                                     StaticMappingQ1<3>::mapping,
                                     source_constraints);
  const auto target_space = fe_space(target_dof_handler,
                                     StaticMappingQ1<3>::mapping,
                                     target_constraints);

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  StateLayout layout;
  const auto  source = source_space.field(layout, "bulk-source");
  const auto  target = target_space.field(layout, "bulk-target");
  Model       model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  (value(source) * test(target) * support).add(builder);

  Vector state;
  state.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
  state = 1.;
  state.compress(VectorOperation::insert);
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);
  Vector                          residual;
  residual.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
  model.evaluate_row(target.field_id(), context, residual);

  const auto matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector action;
  action.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
  matrix->view.vmult(action, state);
  action -= residual;
  EXPECT_LT(action.l2_norm(), 1.e-12);

  double local_measure = 0.;
  for (const auto &point : support.locally_owned_quadrature_points())
    local_measure += point.weight;
  Vector ones;
  ones.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
  ones = 1.;
  ones.compress(VectorOperation::insert);
  EXPECT_NEAR(residual * ones,
              Utilities::MPI::sum(local_measure, MPI_COMM_WORLD),
              1.e-10);
}
