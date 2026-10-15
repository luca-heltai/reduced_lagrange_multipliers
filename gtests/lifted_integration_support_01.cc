#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/lifted_integration_support.h>
#include <immersx/core/observable_lift.h>
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
      lift.section.inclusion_type        = "hyper_cube";
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
  lift.section.inclusion_type        = "hyper_cube";
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

TEST(LiftedIntegrationSupport, BOTH_IntegratesFrozenNonconstantThickness)
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

TEST(LiftedIntegrationSupport,
     BOTH_RotatedRepresentativeLineUsesPhysicalGeometry)
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
  lift.section.inclusion_type        = "hyper_cube";
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
  const auto source_relevant =
    DoFTools::extract_locally_relevant_dofs(source_dof_handler);
  const auto target_relevant =
    DoFTools::extract_locally_relevant_dofs(target_dof_handler);
  AffineConstraints<double> source_constraints;
  source_constraints.reinit(source_dof_handler.locally_owned_dofs(),
                            source_relevant);
  DoFTools::make_hanging_node_constraints(source_dof_handler,
                                          source_constraints);
  source_constraints.close();
  AffineConstraints<double> target_constraints;
  target_constraints.reinit(target_dof_handler.locally_owned_dofs(),
                            target_relevant);
  DoFTools::make_hanging_node_constraints(target_dof_handler,
                                          target_constraints);
  target_constraints.close();
  const auto source_space = fe_space(source_dof_handler,
                                     StaticMappingQ1<3>::mapping,
                                     source_constraints,
                                     source_relevant);
  const auto target_space = fe_space(target_dof_handler,
                                     StaticMappingQ1<3>::mapping,
                                     target_constraints,
                                     target_relevant);

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
  const auto global_measure =
    Utilities::MPI::sum(local_measure, MPI_COMM_WORLD);
  EXPECT_NEAR(global_measure, 1., 1.e-10);
  Vector ones;
  ones.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
  ones = 1.;
  ones.compress(VectorOperation::insert);
  EXPECT_NEAR(residual * ones, global_measure, 1.e-10);
}

TEST(LiftedIntegrationSupport, BOTH_SourceBulkLiftSamplesPhysicalPoints)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> representative_parameters(
    "/Lifted source-bulk domain/");
  representative_parameters.name_of_grid       = "hyper_cube";
  representative_parameters.arguments_for_grid = "-1: 1: false";
  representative_parameters.initial_refinement = 2;
  Domain<1, 3> representative_domain(representative_parameters);
  representative_domain.make_grid();

  TensorProductLift<1, 2, 3, 3> vector_lift("/Lifted source-bulk geometry/");
  vector_lift.section.inclusion_degree      = 0;
  vector_lift.section.inclusion_type        = "hyper_cube";
  vector_lift.section.selected_coefficients = {0};
  vector_lift.section.n_q_points            = 2;
  vector_lift.representative_n_q_points     = 2;
  vector_lift.thickness                     = "0.25";
  const auto support = representative_domain.lift(vector_lift);

  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(1);
  FESystem<3>   bulk_element(FE_Q<3>(2), 3);
  DoFHandler<3> bulk_dof_handler(background);
  bulk_dof_handler.distribute_dofs(bulk_element);
  AffineConstraints<double> bulk_constraints;
  bulk_constraints.close();
  const auto bulk_space =
    fe_space(bulk_dof_handler, StaticMappingQ1<3>::mapping, bulk_constraints);

  FE_Q<1, 3>       line_element(1);
  DoFHandler<1, 3> line_dof_handler(representative_domain.triangulation());
  line_dof_handler.distribute_dofs(line_element);
  AffineConstraints<double> line_constraints;
  line_constraints.close();
  const auto line_space = fe_space(line_dof_handler,
                                   StaticMappingQ1<1, 3>::mapping,
                                   line_constraints);

  StateLayout layout;
  const auto  bulk =
    bulk_space.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line = line_space.field(layout, "line");

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto lifted_line = lift(value(line), vector_lift);
  weak_term(value(bulk), test(lifted_line)).add(builder);

  const auto lift_points = lifted_line.lifted_points();
  ASSERT_FALSE(lift_points.empty());
  const auto direction = lift_points.front().mode_vector;
  ASSERT_NEAR(direction.norm(), 1., 1.e-12);

  class TransverseQuadratic : public Function<3>
  {
  public:
    explicit TransverseQuadratic(const Tensor<1, 3> &direction)
      : Function<3>(3)
      , direction_(direction)
    {}

    double
    value(const Point<3>    &point,
          const unsigned int component = 0) const override
    {
      return direction_[component] *
             (1. + point[1] * point[1] + point[2] * point[2]);
    }

  private:
    const Tensor<1, 3> direction_;
  } function(direction);

  Vector bulk_values;
  bulk_values.reinit(bulk.locally_owned_dofs(), MPI_COMM_WORLD);
  VectorTools::interpolate(bulk_dof_handler, function, bulk_values);

  StateView<Vector> state_view(layout, 0.);
  state_view.bind(bulk.field_id(), bulk_values);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      matrix =
    model.state_matrix_operator(line.field_id(), bulk.field_id(), context);
  ASSERT_TRUE(matrix.has_value());

  Vector line_ones;
  line_ones.reinit(line.locally_owned_dofs(), MPI_COMM_WORLD);
  line_ones = 1.;
  line_ones.compress(VectorOperation::insert);
  Vector coupled_values;
  coupled_values.reinit(line.locally_owned_dofs(), MPI_COMM_WORLD);
  matrix->view.vmult(coupled_values, bulk_values);
  const double coupled_integral = line_ones * coupled_values;

  // Independently integrate the exact bulk polynomial at the support's
  // physical points. The representative curve has zero transverse values.
  double      local_physical_integral       = 0.;
  double      local_representative_integral = 0.;
  const auto &support_points = support.locally_owned_quadrature_points();
  ASSERT_EQ(lift_points.size(), support_points.size());
  for (std::size_t q = 0; q < support_points.size(); ++q)
    {
      const auto &point        = support_points[q];
      const auto &lifted_point = lift_points[q];
      EXPECT_EQ(point.stable_id, lifted_point.stable_id);
      EXPECT_NEAR(point.weight, lifted_point.weight, 1.e-12);
      EXPECT_NEAR(direction * lifted_point.mode_vector, 1., 1.e-12);
      local_physical_integral += point.weight *
                                 (direction * lifted_point.mode_vector) *
                                 (1. + point.point[1] * point.point[1] +
                                  point.point[2] * point.point[2]);
      local_representative_integral +=
        point.weight * (direction * lifted_point.mode_vector) *
        (1. + point.representative_point[1] * point.representative_point[1] +
         point.representative_point[2] * point.representative_point[2]);
    }
  const double physical_integral =
    Utilities::MPI::sum(local_physical_integral, MPI_COMM_WORLD);
  const double representative_integral =
    Utilities::MPI::sum(local_representative_integral, MPI_COMM_WORLD);
  EXPECT_GT(physical_integral, representative_integral + 1.e-6);
  EXPECT_NEAR(coupled_integral, physical_integral, 1.e-10);
}

TEST(LiftedIntegrationSupport,
     BOTH_SelectedVectorModesPreserveAmplitudeAndRegistration)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  FiniteElementSpace<1, 3>      space(fixture.domain.triangulation(),
                                 std::make_unique<FE_Q<1, 3>>(1));
  const auto                    view = space.view();
  TensorProductLift<1, 2, 3, 3> geometry("/Selected vector modes/");
  geometry.thickness                     = "0.1";
  geometry.section.inclusion_degree      = 1;
  geometry.section.selected_coefficients = {1, 4, 7};
  const auto observable =
    lift(value(view.field("coefficient")), geometry).with_id(FieldId(42));
  ReferenceCrossSection<1, 3, 3> section(geometry.section);
  Tensor<1, 3>                   tangent;
  tangent[0]           = 1.;
  unsigned int nonunit = 0;
  for (unsigned int slot = 0; slot < 3; ++slot)
    {
      const auto selected = observable.mode(slot);
      EXPECT_EQ(selected.source().id(), FieldId(42));
      ASSERT_EQ(selected.dependencies().size(), 1u);
      EXPECT_EQ(selected.dependencies().front(), FieldId(42));
      for (const auto &point : selected.lifted_points())
        {
          const auto expected =
            section.get_transformed_mode_values(point.section_qpoint, tangent);
          ASSERT_EQ(point.mode_values.size(), 3u);
          for (unsigned int d = 0; d < 3; ++d)
            {
              EXPECT_NEAR(point.mode_vector[d], expected[3 * slot + d], 1.e-14);
              EXPECT_NEAR(point.mode_values[d], expected[3 * slot + d], 1.e-14);
            }
          if (std::abs(point.mode_vector.norm() - 1.) > 1.e-4)
            ++nonunit;
        }
    }
  EXPECT_GT(Utilities::MPI::sum(nonunit, MPI_COMM_WORLD), 0u);
  EXPECT_THROW(observable.mode(3), ExceptionBase);
}

TEST(LiftedIntegrationSupport,
     BOTH_RejectsChangedStaticAffineConstraintsCollectively)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  TensorProductLift<1, 2, 3, 3> geometry("/Static affine lift/");
  geometry.thickness                     = "0.1";
  geometry.section.selected_coefficients = {0};
  FiniteElementSpace<1, 3> line_space(fixture.domain.triangulation(),
                                      std::make_unique<FE_Q<1, 3>>(1));
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, -2., 2.);
  mesh.refine_global(1);
  FiniteElementSpace<3> bulk_space(mesh,
                                   std::make_unique<FESystem<3>>(FE_Q<3>(1),
                                                                 3));
  auto                 &constraints = bulk_space.constraints();
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  if (bulk_space.locally_relevant_dofs().is_element(0))
    constraints.add_line(0);
  constraints.close();
  const auto  bulk_view = bulk_space.view();
  const auto  line_view = line_space.view();
  StateLayout layout;
  const auto  bulk =
    bulk_view.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line = line_view.field(layout, "line");
  using V         = ImmersXLA::MPI::Vector;
  using M         = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(bulk), test(lift(value(line), geometry).mode(0)))
    .add(builder);
  V state, residual;
  detail::initialize_weak_vector(state,
                                 bulk.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(residual,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  state = 1.;
  StateView<V> view(layout, 0.);
  view.bind(bulk.id(), state);
  EvaluationContext<V> context(0., view);
  EXPECT_NO_THROW(model.evaluate_row(line.id(), context, residual));
  if (bulk.locally_owned_dofs().is_element(0))
    constraints.set_inhomogeneity(0, 0.1);
  EXPECT_THROW(model.evaluate_row(line.id(), context, residual), ExceptionBase);
}

TEST(LiftedIntegrationSupport, BOTH_VectorLiftHangingNodesAndAffineSourceShift)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  TensorProductLift<1, 2, 3, 3> geometry("/Hanging affine vector lift/");
  geometry.thickness                     = "0.1";
  geometry.section.selected_coefficients = {0};
  FiniteElementSpace<1, 3> line_space(fixture.domain.triangulation(),
                                      std::make_unique<FE_Q<1, 3>>(1));
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, -1., 1.);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 0.)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  FiniteElementSpace<3> bulk_space(mesh,
                                   std::make_unique<FESystem<3>>(FE_Q<3>(1),
                                                                 3));
  auto                 &constraints = bulk_space.constraints();
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  DoFTools::make_hanging_node_constraints(bulk_space.dof_handler(),
                                          constraints);
  EXPECT_GT(Utilities::MPI::sum(constraints.n_constraints(), MPI_COMM_WORLD),
            0u);
  Functions::ConstantFunction<3> exact(std::vector<double>{0.1, -0.2, 0.3});
  VectorTools::interpolate_boundary_values(bulk_space.dof_handler(),
                                           0,
                                           exact,
                                           constraints);
  constraints.close();
  const auto  bulk_view = bulk_space.view();
  const auto  line_view = line_space.view();
  StateLayout layout;
  const auto  bulk =
    bulk_view.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line   = line_view.field(layout, "line");
  const auto lifted = lift(value(line), geometry).mode(0);
  using V           = ImmersXLA::MPI::Vector;
  using M           = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(bulk), test(lifted)).add(builder);
  V state, residual, expected;
  detail::initialize_weak_vector(state,
                                 bulk.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(residual,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(expected,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  VectorTools::interpolate(bulk.dof_handler(), exact, state);
  constraints.distribute(state);
  StateView<V> view(layout, 0.);
  view.bind(bulk.id(), state);
  EvaluationContext<V> context(0., view);
  model.evaluate_row(line.id(), context, residual);
  // Integrate the exact physical constant, including contributions whose
  // source shape functions have been eliminated by inhomogeneous constraints.
  Tensor<1, 3> constant;
  constant[0] = 0.1;
  constant[1] = -0.2;
  constant[2] = 0.3;
  for (const auto &point : lifted.lifted_points())
    {
      dealii::Vector<double> local(point.dof_indices.size());
      for (unsigned int i = 0; i < local.size(); ++i)
        local[i] = point.weight * point.source_basis_values[i] *
                   (constant * point.mode_vector);
      line.constraints().distribute_local_to_global(local,
                                                    point.dof_indices,
                                                    expected);
    }
  expected.compress(VectorOperation::add);
  EXPECT_GT(expected.l2_norm(), 1.e-5);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);
}

TEST(LiftedIntegrationSupport,
     BOTH_VectorLiftExplicitAffineUpdatesAndStructureGuard)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  TensorProductLift<1, 2, 3, 3> geometry("/Updated vector affine lift/");
  geometry.thickness                     = "0.1";
  geometry.section.selected_coefficients = {0};
  FiniteElementSpace<1, 3> line_space(fixture.domain.triangulation(),
                                      std::make_unique<FE_Q<1, 3>>(1));
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, -2., 2.);
  mesh.refine_global(1);
  FiniteElementSpace<3> bulk_space(mesh,
                                   std::make_unique<FESystem<3>>(FE_Q<3>(1),
                                                                 3));
  auto                 &constraints = bulk_space.constraints();
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  for (const auto i : bulk_space.locally_relevant_dofs())
    constraints.add_line(i);
  constraints.close();
  const auto  bulk_view = bulk_space.view();
  const auto  line_view = line_space.view();
  StateLayout layout;
  const auto  bulk =
    bulk_view.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line   = line_view.field(layout, "line");
  const auto lifted = lift(value(line), geometry).mode(0);
  using V           = ImmersXLA::MPI::Vector;
  using M           = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(bulk), test(lifted))
    .with_lifted_affine_updates()
    .add(builder);
  V state, residual, expected;
  detail::initialize_weak_vector(state,
                                 bulk.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(residual,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(expected,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  state = 0.;
  StateView<V> view(layout, 0.);
  view.bind(bulk.id(), state);
  EvaluationContext<V> context(0., view);
  model.evaluate_row(line.id(), context, residual);
  EXPECT_DOUBLE_EQ(residual.l2_norm(), 0.);
  Tensor<1, 3> constant;
  constant[0] = 0.1;
  constant[1] = -0.2;
  constant[2] = 0.3;
  for (const auto &cell : bulk.dof_handler().active_cell_iterators())
    if (!cell->is_artificial())
      {
        std::vector<types::global_dof_index> indices(
          cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(indices);
        for (unsigned int i = 0; i < indices.size(); ++i)
          constraints.set_inhomogeneity(
            indices[i],
            constant[cell->get_fe().system_to_component_index(i).first]);
      }
  EXPECT_NO_THROW(model.evaluate_row(line.id(), context, residual));
  for (const auto &point : lifted.lifted_points())
    {
      dealii::Vector<double> local(point.dof_indices.size());
      for (unsigned int i = 0; i < local.size(); ++i)
        local[i] = point.weight * point.source_basis_values[i] *
                   (constant * point.mode_vector);
      line.constraints().distribute_local_to_global(local,
                                                    point.dof_indices,
                                                    expected);
    }
  expected.compress(VectorOperation::add);
  EXPECT_GT(expected.l2_norm(), 1.e-5);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);
  // Offsets may change, but an already condensed matrix cannot follow a new
  // set of constraint masters or constrained DoFs.
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  constraints.close();
  EXPECT_THROW(model.evaluate_row(line.id(), context, residual), ExceptionBase);
}
