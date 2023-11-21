#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "inclusions.h"

using namespace dealii;

TEST(TestInclusion2, CheckPoints) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back({{0, 0, 1.0}});
  ref.initialize();
  const auto &p = ref.get_current_support_points(0);

  ASSERT_NEAR(p[0].distance(Point<2>(1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[1].distance(Point<2>(0, 1)), 0, 1e-10);
  ASSERT_NEAR(p[2].distance(Point<2>(-1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[3].distance(Point<2>(0, -1)), 0, 1e-10);
}

TEST(TestInclusion3, CheckPoints) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 0, 0, 1.0, 1.0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto &p = ref.get_current_support_points(0);

  ASSERT_NEAR(p[0].distance(Point<3>(1, 0, 0)), 0, 1e-10);
  ASSERT_NEAR(p[1].distance(Point<3>(0, 1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[2].distance(Point<3>(-1, 0, 0)), 0, 1e-10);
  ASSERT_NEAR(p[3].distance(Point<3>(0, -1, 0)), 0, 1e-10);
}

TEST(TestInclusion3, CheckPointsRotated) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 1.0, 0, 0, 1.0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto &p = ref.get_current_support_points(0);

  ASSERT_NEAR(p[0].norm(), 1, 1e-10);
  ASSERT_NEAR(p[1].norm(), 1, 1e-10);
  ASSERT_NEAR(p[2].norm(), 1, 1e-10);
  ASSERT_NEAR(p[3].norm(), 1, 1e-10);

  ASSERT_NEAR(p[0].distance(Point<3>(0, 0, -1)), 0, 1e-10);
  ASSERT_NEAR(p[1].distance(Point<3>(0, 1, 0)), 0, 1e-10);
  ASSERT_NEAR(p[2].distance(Point<3>(0, 0, 1)), 0, 1e-10);
  ASSERT_NEAR(p[3].distance(Point<3>(0, -1, 0)), 0, 1e-10);
}


TEST(TestInclusion3, CheckNegativeZDirection) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 0, 0, -1, 1.0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto r = ref.get_rotation(0);

  const Tensor<1, 3> zdir({0, 0, 1});
  const Tensor<1, 3> mzdir({0, 0, -1});

  const auto a = r * zdir;

  ASSERT_NEAR((a - mzdir).norm(), 0, 1e-10);
}

TEST(TestInclusion3, CheckAlmostNegativeZDirection) // NOLINT
{
  // cx, cy, cz, dx, dy, dz, r
  std::vector<double> inc({{0, 0, 0, 0, 0.5, -1, 1.0}});
  Inclusions<3>       ref;
  ref.set_n_q_points(4);
  ref.set_n_coefficients(1);
  ref.inclusions.push_back(inc);
  ref.initialize();
  const auto r = ref.get_rotation(0);

  const Tensor<1, 3> zdir({0, 0, 1});
  Tensor<1, 3>       mzdir({0, 0.5, -1});
  mzdir /= mzdir.norm();

  const auto a = r * zdir;

  ASSERT_NEAR((a - mzdir).norm(), 0, 1e-10);
}