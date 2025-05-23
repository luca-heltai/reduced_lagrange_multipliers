// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "inclusions.h"
#include "reference_cross_section.h" // Add include for ReferenceCrossSection

using namespace dealii;

TEST(ReferenceCrossSection, CheckBasisOrthogonality) // NOLINT
{
  const int          dim          = 2;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Or "hyper_cube"
  par.refinement_level = 2;            // Use a reasonable refinement level
  // Leaving par.selected_coefficients empty means all basis functions are
  // selected

  // Create the reference inclusion object
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the computed basis functions and the mass matrix
  const auto &basis       = ref_inclusion.get_basis_functions();
  const auto &mass_matrix = ref_inclusion.get_mass_matrix();

  const unsigned int n_basis_functions = basis.size();

  // Verify the number of basis functions matches the polynomial space size
  PolynomialsP<spacedim> polynomials(degree);
  ASSERT_EQ(n_basis_functions, polynomials.n() * n_components);

  // Check for M-orthonormality: basis[i]^T * M * basis[j] == delta_ij
  Vector<double> tmp(mass_matrix.m()); // Temporary vector for M * basis[j]

  for (unsigned int i = 0; i < n_basis_functions; ++i)
    {
      ASSERT_GT(basis[i].l2_norm(), 1e-15)
        << "Basis function " << i << " is zero.";
      for (unsigned int j = 0; j < n_basis_functions; ++j)
        {
          const double dot_product =
            mass_matrix.matrix_scalar_product(basis[i], basis[j]);

          if (i == j)
            {
              // Diagonal elements should be close to |D| for orthogonal basis
              ASSERT_NEAR(dot_product, ref_inclusion.measure(), 1e-10)
                << "Basis functions " << i << " and " << j
                << " are not M-orthonormal (diagonal check).";
            }
          else
            {
              // Off-diagonal elements should be close to 0 for orthogonal basis
              ASSERT_NEAR(dot_product, 0.0, 1e-10)
                << "Basis functions " << i << " and " << j
                << " are not M-orthogonal (off-diagonal check).";
            }
        }
    }
}


TEST(ReferenceCrossSection, CheckDiskQuadrature) // NOLINT
{
  const int          dim          = 1;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Use a circular domain
  par.refinement_level = 5;            // Use a reasonable refinement level

  // Create the reference inclusion object
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the global quadrature
  const auto &quadrature = ref_inclusion.get_global_quadrature();

  // Compute the sum of all quadrature weights, which should equal
  // the measure of the domain (2*pi for a unit circle, with r=1)
  double sum = 0.0;
  for (const auto &weight : quadrature.get_weights())
    {
      sum += weight;
    }

  // The area of a unit circle is 2*pi
  ASSERT_NEAR(sum, 2 * numbers::PI, 1e-3)
    << "Integral of unit function over unit circle should equal 2*pi";

  // Test that a constant function integrates as expected
  double const_integral = 0.0;
  double const_value    = 2.5; // Some arbitrary constant

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const_integral += const_value * quadrature.weight(q);
    }

  ASSERT_NEAR(const_integral, const_value * 2 * numbers::PI, 2e-3)
    << "Constant function integral incorrect";

  // Test that x^2 + y^2 integrates to expected value over unit circle
  // For x^2 + y^2 over unit circle, the exact result is pi/2
  double r_squared_integral = 0.0;

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const Point<spacedim> &p = quadrature.point(q);
      r_squared_integral += p.square() * quadrature.weight(q);
    }

  ASSERT_NEAR(r_squared_integral, 2 * numbers::PI, 1e-3)
    << "Integral of r^2 over unit circle should equal 2 pi";
}

TEST(ReferenceCrossSection, CheckCircleQuadrature) // NOLINT
{
  const int          dim          = 2;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Use a circular domain
  par.refinement_level = 5;            // Use a reasonable refinement level

  // Create the reference inclusion object
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the global quadrature
  const auto &quadrature = ref_inclusion.get_global_quadrature();

  // Compute the sum of all quadrature weights, which should equal
  // the measure of the domain (pi*r^2 for a unit circle, with r=1)
  double sum = 0.0;
  for (const auto &weight : quadrature.get_weights())
    {
      sum += weight;
    }

  // The area of a unit circle is pi
  ASSERT_NEAR(sum, numbers::PI, 1e-2)
    << "Integral of unit function over unit circle should equal pi";

  // Test that a constant function integrates as expected
  double const_integral = 0.0;
  double const_value    = 2.5; // Some arbitrary constant

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const_integral += const_value * quadrature.weight(q);
    }

  ASSERT_NEAR(const_integral, const_value * numbers::PI, 1e-2)
    << "Constant function integral incorrect";

  // Test that x^2 + y^2 integrates to expected value over unit circle
  // For x^2 + y^2 over unit circle, the exact result is pi/2
  double r_squared_integral = 0.0;

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const Point<spacedim> &p         = quadrature.point(q);
      double                 r_squared = p.norm_square(); // x^2 + y^2
      r_squared_integral += r_squared * quadrature.weight(q);
    }

  ASSERT_NEAR(r_squared_integral, numbers::PI / 2.0, 1e-2)
    << "Integral of r^2 over unit circle should equal pi/2";
}

TEST(ReferenceCrossSection, Check3DBallQuadrature) // NOLINT
{
  const int          dim          = 3;
  const int          spacedim     = 3;
  const int          n_components = 1;
  const unsigned int degree       = 2;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Use a 3D ball domain
  par.refinement_level = 2;            // Use a reasonable refinement level

  // Create the reference inclusion object
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the global quadrature
  const auto &quadrature = ref_inclusion.get_global_quadrature();

  // Compute the sum of all quadrature weights, which should equal
  // the measure of the domain (4/3*pi*r^3 for a unit ball, with r=1)
  double sum = 0.0;
  for (const auto &weight : quadrature.get_weights())
    {
      sum += weight;
    }

  // The volume of a unit ball is 4π/3
  const double unit_ball_volume = 4.0 * numbers::PI / 3.0;
  ASSERT_NEAR(sum, unit_ball_volume, .3)
    << "Integral of unit function over unit 3D ball should equal 4π/3";

  // Test that a constant function integrates as expected
  double const_integral = 0.0;
  double const_value    = 2.5; // Some arbitrary constant

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const_integral += const_value * quadrature.weight(q);
    }

  ASSERT_NEAR(const_integral, const_value * unit_ball_volume, 0.7)
    << "Constant function integral incorrect";

  // Test that x^2 + y^2 + z^2 integrates to expected value over unit ball
  // For r^2 over unit ball, the exact result is 4π/5
  double r_squared_integral = 0.0;

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const Point<spacedim> &p         = quadrature.point(q);
      double                 r_squared = p.norm_square(); // x^2 + y^2 + z^2
      r_squared_integral += r_squared * quadrature.weight(q);
    }

  const double expected_r_squared = 4.0 * numbers::PI / 5.0;
  ASSERT_NEAR(r_squared_integral, expected_r_squared, 0.3)
    << "Integral of r^2 over unit 3D ball should equal 4π/5";
}

TEST(ReferenceCrossSection, CheckRotatedDisk) // NOLINT
{
  const int          dim          = 1;
  const int          spacedim     = 3;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Use a circular domain
  par.refinement_level = 1;            // Use a reasonable refinement level

  // Create the reference inclusion object for the original disk
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the global quadrature for the reference disk
  const auto &quadrature = ref_inclusion.get_global_quadrature();

  // Compute the sum of all quadrature weights for the reference disk
  double reference_sum = 0.0;
  for (const auto &weight : quadrature.get_weights())
    {
      reference_sum += weight;
    }

  auto rotated_quadrature =
    ref_inclusion.get_transformed_quadrature(Point<3>(),
                                             Tensor<1, 3>({0, 1, 0}),
                                             1.0);

  // Check that the sum of weights is the same (measure of domain is preserved)
  double rotated_sum = 0.0;
  for (const auto &weight : rotated_quadrature.get_weights())
    {
      rotated_sum += weight;
    }

  std::cout << "Reference sum: " << reference_sum
            << ", Rotated sum: " << rotated_sum << std::endl;

  ASSERT_NEAR(rotated_sum, reference_sum, 1e-10)
    << "Measure of rotated disk should equal reference disk";

  // Check that for each quadrature point, the x coordinate is approximately 0
  // (since the disk is now in the y-z plane)
  for (unsigned int q = 0; q < rotated_quadrature.size(); ++q)
    {
      const Point<spacedim> &p = rotated_quadrature.point(q);
      ASSERT_NEAR(p[1], 0.0, 1e-10)
        << "Y-coordinate of point " << p << " should be zero after rotation";

      // Verify that the sum of squares of the non-zero coordinates is ≤ 1
      // (points should still be inside the unit disk, now in y-z plane)
      ASSERT_NEAR(p.square(), 1.0, 1e-3)
        << "Rotated point " << q << " lies outside the unit disk";
    }
}



TEST(ReferenceCrossSection, CheckP1) // NOLINT
{
  const int          dim          = 1;
  const int          spacedim     = 3;
  const int          n_components = 1;
  const unsigned int degree       = 1;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Use a circular domain
  par.refinement_level = 1;            // Use a reasonable refinement level

  // Create the reference inclusion object for the original disk
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  ASSERT_EQ(ref_inclusion.max_n_basis(), 3);
  ASSERT_EQ(ref_inclusion.n_selected_basis(), 3)
    << "P1 inclusion should have 3 selected basis functions";
}

TEST(ReferenceCrossSection, CheckMeasure) // NOLINT
{
  // Test case 1: 1D inclusion in 2D space (Disk, effectively a line segment)
  {
    const int          dim          = 1;
    const int          spacedim     = 2;
    const int          n_components = 1;
    const unsigned int degree = 2; // Degree doesn't affect measure directly

    ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
    par.inclusion_degree = degree;
    par.inclusion_type   = "hyper_ball";
    par.refinement_level = 3; // Sufficient refinement for measure

    ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

    // Check non-scaled measure (length of the boundary of a unit disk is 2*pi)
    double expected_measure_non_scaled = 2.0 * numbers::PI;
    ASSERT_NEAR(ref_inclusion.measure(), expected_measure_non_scaled, 1e-5)
      << "Non-scaled measure for 1D hyper_ball (dim=1, spacedim=2) is incorrect.";

    // Check scaled measure
    const double scale = 2.5;
    double       expected_measure_scaled =
      expected_measure_non_scaled * std::pow(scale, dim);
    ASSERT_NEAR(ref_inclusion.measure(scale), expected_measure_scaled, 1e-4)
      << "Scaled measure for 1D hyper_ball (dim=1, spacedim=2) is incorrect.";
  }

  // Test case 2: 2D inclusion in 2D space (Circle)
  {
    const int          dim          = 2;
    const int          spacedim     = 2;
    const int          n_components = 1;
    const unsigned int degree = 2; // Degree doesn't affect measure directly

    ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
    par.inclusion_degree = degree;
    par.inclusion_type   = "hyper_ball";
    par.refinement_level = 5; // Higher refinement for better area approximation

    ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

    // Check non-scaled measure (area of a unit circle is pi)
    double expected_measure_non_scaled = numbers::PI;
    ASSERT_NEAR(ref_inclusion.measure(),
                expected_measure_non_scaled,
                1e-2) // Tolerance might need adjustment based on refinement
      << "Non-scaled measure for 2D hyper_ball (circle) is incorrect.";

    // Check scaled measure
    const double scale = 1.5;
    double       expected_measure_scaled =
      expected_measure_non_scaled * std::pow(scale, dim);
    ASSERT_NEAR(ref_inclusion.measure(scale), expected_measure_scaled, 1e-2)
      << "Scaled measure for 2D hyper_ball (circle) is incorrect.";
  }
}
