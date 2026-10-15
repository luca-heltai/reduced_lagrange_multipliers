// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_lib.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/boundary_conditions.h>
#include <immersx/core/domain.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/known_term.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/modal_lift.h>

using namespace ImmersX;
using namespace dealii;

TEST(LiftedPoisson, BOTH_NonzeroBoundaryAndMultipleModes)
{
  ParameterAcceptor::clear();
  DomainParameters<3> dp("/Lifted test/Bulk/");
  dp.arguments_for_grid = "-1:1:false";
  dp.initial_refinement = 2;
  Domain<3> bulk(dp);
  bulk.make_grid();
  DomainParameters<1, 3> rp("/Lifted test/Curve/");
  Domain<1, 3>           curve(rp);
  curve.prepare_grid(true);
  curve.refinement_triangulation().clear();
  std::vector<CellData<1>> cells(2);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 1;
  cells[1].vertices[0] = 1;
  cells[1].vertices[1] = 2;
  curve.refinement_triangulation().create_triangulation({Point<3>(0., 0., -.5),
                                                         Point<3>(0., 0., 0.),
                                                         Point<3>(0., 0., .5)},
                                                        cells,
                                                        SubCellData());
  curve.finalize_grid();
  FiniteElementSpaceParameters<3>    fp("/Lifted test/Bulk space/");
  FiniteElementSpaceParameters<1, 3> mp("/Lifted test/Modal space/",
                                        "FESystem<1,3>[FE_Q<1,3>(1)^3]");
  FiniteElementSpace<3>              fs(bulk.triangulation(), fp);
  FiniteElementSpace<1, 3>           ms(curve.triangulation(), mp);
  BoundaryConditions<3>              bc(fs);
  Functions::ConstantFunction<3>     boundary(2.);
  bc.add_dirichlet(0, boundary);
  bc.update(0.);
  auto                    view = fs.view();
  auto                    mv   = ms.view();
  LinearAdapterParameters options("/Lifted test/Solver/");
  options.solver = LinearSolver::direct;
  LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
    options, MPI_COMM_WORLD);
  const auto u = adapter.add(algebraic_field(view.field("u"))).fields();
  adapter.add(weak_term(gradient(u), gradient(test(u))));
  TensorProductLift<1, 2, 3> lift("/Lifted test/Lift/");
  lift.thickness                     = "0.2";
  lift.section.inclusion_degree      = 1;
  lift.section.selected_coefficients = {0, 1, 2};
  auto geometry                      = modal_lift(mv.field("lambda"), lift);
  // The constant trace has modal coefficient 2 in the constant mode and zero
  // coefficients in the two degree-one modes. Nonhomogeneous BC enter B u.
  auto data = std::make_shared<Functions::ConstantFunction<3>>(
    std::vector<double>{2., 0., 0.});
  const auto lambda =
    adapter.add(LiftedMultiplierInteraction(u, geometry, data))
      .fields()
      .multiplier;
  auto state = adapter.make_state();
  adapter.solve(state);
  auto residual = adapter.make_state();
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-9);
  EXPECT_LT(adapter.field(state, lambda).l2_norm(), 1.e-8);
  double error = 0.;
  for (auto i : adapter.field(state, u).locally_owned_elements())
    error = std::max(error, std::abs(adapter.field(state, u)[i] - 2.));
  EXPECT_LT(Utilities::MPI::max(error, MPI_COMM_WORLD), 1.e-9);
  boundary.set_time(1.);
  // Refresh the same persistent constraints after assembly, without rebuilding
  // B. The modal row must see the changed affine trace as well as the bulk row.
  Functions::ConstantFunction<3> changed_boundary(3.);
  bc.clear();
  bc.add_dirichlet(0, changed_boundary);
  bc.update(0.);
  adapter.evaluate_residual(state, residual);
  EXPECT_GT(residual.l2_norm(), 1.e-3);
  adapter.solve(state);
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-9);
}
