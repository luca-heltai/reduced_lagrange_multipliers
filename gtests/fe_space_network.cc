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

#include <deal.II/base/exception_macros.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include "immersed_repartitioner.h"
#include "particle_coupling.h"
#include "tensor_product_space.h"
#include "utils.h"

TEST(FESpaceOnNetwork, SanityChecks) // NOLINT
{
  static constexpr int dim          = 1;
  static constexpr int spacedim     = 3;
  static constexpr int n_components = 1;

  Triangulation<spacedim, spacedim> space_tria;
  GridGenerator::hyper_cube(space_tria, -.4, 1.2);
  space_tria.refine_global(4);

  const std::string     filename = SOURCE_DIR "/data/tests/mstree_100.vtk";
  GridIn<dim, spacedim> gridin;
  Triangulation<dim, spacedim> immersed_tria;
  gridin.attach_triangulation(immersed_tria);
  std::ifstream input_file(filename);
  gridin.read_vtk(input_file);

  DoFHandler<dim, spacedim> dof_handler_network(immersed_tria);
  FE_Q<dim, spacedim>       fe(1);
  dof_handler_network.distribute_dofs(fe);
  std::cout << "Number of DoFs: " << dof_handler_network.n_dofs() << std::endl;


  // Lambda for interpolation test at junctions
  auto test_interpolation_at_junctions =
    [](const DoFHandler<dim, spacedim> &dof_handler) {
      static_assert(dim == 1 && spacedim == 3,
                    "This function only works for 1D-3D trias.");

      const auto &tria                       = dof_handler.get_triangulation();
      const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();

      std::cout << "========== INTERPOLATION TEST AT JUNCTIONS =========="
                << std::endl;
      std::cout << "Using " << fe.get_name() << " elements" << std::endl;

      // Get non-manifold faces
      const auto junction_faces = get_non_manifold_faces(tria);
      std::cout << "Found " << junction_faces.size() << " junctions to test"
                << std::endl;

      if (junction_faces.empty())
        {
          std::cout << "No junctions found. Return." << std::endl;
          return;
        }

      TestFunction func;

      // Interpolate function to FE space
      Vector<double> interpolated(dof_handler.n_dofs());
      VectorTools::interpolate(dof_handler, func, interpolated);

      // Check at each junction
      unsigned int continuity_issues = 0;
      unsigned int accuracy_issues   = 0;

      for (const auto &[face, insisting_cells] : junction_faces)
        {
          // Get junction coordinates
          const Point<spacedim> junction_point = face->center();
          const double          expected_value = func.value(junction_point);

          std::cout << "  Junction at " << junction_point
                    << ", expected value: " << expected_value << std::endl;

          // Collect values at this junction from all connecting cells
          std::map<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                   std::vector<double>>
                           cell_values;
          std::set<double> unique_values;

          for (const auto &cell : insisting_cells)
            {
              // Find which vertex of this cell is at the junction (we have only
              // 2 vertices to test)
              bool         vertex0_at_junction = (cell->face(0) == face);
              unsigned int junction_vertex_idx = vertex0_at_junction ? 0 : 1;

              // Get DoF indices for this cell
              typename DoFHandler<dim, spacedim>::active_cell_iterator dof_cell(
                &tria, cell->level(), cell->index(), &dof_handler);

              std::vector<types::global_dof_index> dof_indices(
                fe.dofs_per_cell);
              dof_cell->get_dof_indices(dof_indices);

              // Get values at the junction vertex
              std::vector<double> values;
              for (unsigned int j = 0; j < fe.dofs_per_vertex; ++j)
                {
                  const unsigned int dof_idx_in_cell =
                    fe.dofs_per_vertex * junction_vertex_idx + j;
                  double value = interpolated[dof_indices[dof_idx_in_cell]];
                  values.push_back(value);
                  unique_values.insert(value);
                }

              cell_values[dof_cell] = values;
            }

          // Output values for each cell
          std::cout << "  Values by cell:" << std::endl;
          for (const auto &[cell, values] : cell_values)
            {
              std::cout << "    Cell " << cell->index() << ": ";
              for (double val : values)
                std::cout << val << " ";
              std::cout << std::endl;
            }

          // Check continuity - are all values the same?
          if (unique_values.size() > 1)
            {
              std::cout << "  CONTINUITY ERROR: Multiple values at junction!"
                        << std::endl;
              std::cout << "  Unique values: ";
              for (double val : unique_values)
                std::cout << val << " ";
              std::cout << std::endl;
              continuity_issues++;
            }
          else
            {
              std::cout << "  Continuity check: PASSED" << std::endl;
            }

          // Check accuracy - is the value correct?
          double actual_value = *unique_values.begin();
          double error        = std::abs(actual_value - expected_value);

          std::cout << "  Interpolation error: " << error << std::endl;

          if (error > tolerance)
            {
              std::cout << "  ACCURACY ERROR: Large interpolation error!"
                        << std::endl;
              accuracy_issues++;
            }
          else
            {
              std::cout << "  Accuracy check: PASSED" << std::endl;
            }
        }

      // Summary for this test function
      std::cout << "  Continuity issues: " << continuity_issues << " out of "
                << junction_faces.size() << " junctions" << std::endl;
      std::cout << "  Accuracy issues: " << accuracy_issues << " out of "
                << junction_faces.size() << " junctions" << std::endl;

      if (continuity_issues == 0 && accuracy_issues == 0)
        std::cout << "  All tests PASSED" << std::endl;
      else
        std::cout << "  Some tests FAILED" << std::endl;


      std::cout << "===================================================="
                << std::endl;

      ASSERT_EQ(continuity_issues, 0)
        << "Continuity issues found at junctions.";
      ASSERT_EQ(accuracy_issues, 0) << "Accuracy issues found at junctions.";
    };

  // Lambda for DoF distribution test at junctions
  auto test_DoF_at_junctions = [](
                                 const DoFHandler<dim, spacedim> &dof_handler) {
    // 2. Analyze DoF distribution at junctions
    std::cout << "\nDOF DISTRIBUTION AT JUNCTIONS:" << std::endl;

    const auto                         &tria = dof_handler.get_triangulation();
    const FiniteElement<dim, spacedim> &fe   = dof_handler.get_fe();

    // Get non-manifold faces
    const auto junction_faces = get_non_manifold_faces(tria);
    std::cout << "Found " << junction_faces.size() << " junctions to test"
              << std::endl;

    if (junction_faces.empty())
      {
        std::cout << "No junctions found. Return." << std::endl;
        return;
      }


    for (const auto &[face, cells] : junction_faces)
      {
        const Point<3> junction_point = face->center();
        std::cout << "Junction at " << junction_point << ":" << std::endl;

        // Collect all DoFs at this junction
        std::map<types::global_dof_index, unsigned int> junction_dof_usage;

        for (const auto &cell : cells)
          {
            // Find which vertex of the cell is at the junction
            bool         vertex0_at_junction = (cell->face(0) == face);
            unsigned int junction_vertex_idx = vertex0_at_junction ? 0 : 1;

            // Get corresponding DoF handler cell
            typename DoFHandler<dim, spacedim>::active_cell_iterator dof_cell(
              &tria, cell->level(), cell->index(), &dof_handler);

            std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
            dof_cell->get_dof_indices(dof_indices);

            // Count DoFs at junction vertex
            for (unsigned int j = 0; j < fe.dofs_per_vertex; ++j)
              {
                const unsigned int dof_idx_in_cell =
                  fe.dofs_per_vertex * junction_vertex_idx + j;
                junction_dof_usage[dof_indices[dof_idx_in_cell]]++;
              }
          }

        // Analyze DoF sharing pattern
        std::map<unsigned int, unsigned int> dof_sharing_histogram;
        for (const auto &[dof_idx, usage_count] : junction_dof_usage)
          dof_sharing_histogram[usage_count]++;

        for (const auto &[count, frequency] : dof_sharing_histogram)
          {
            std::cout << "  " << frequency << " DoFs shared by " << count
                      << " cells" << std::endl;
          }

        // DoFs should be shared across all cells at junction
        if (dof_sharing_histogram.rbegin()->first < cells.size())
          std::cout << "  No DoFs shared by all cells at this junction!"
                    << std::endl;

        // Show the actual DoF indices
        std::cout << "  DoF indices at junction:" << std::endl;
        for (const auto &[dof_idx, count] : junction_dof_usage)
          std::cout << "    DoF " << dof_idx << " used by " << count << " cells"
                    << std::endl;

        AssertThrow(false,
                    ExcMessage("No DoFs shared by all cells at this junction!"))
      }
  };

  // Run the tests
  test_interpolation_at_junctions(dof_handler_network);
  test_DoF_at_junctions(dof_handler_network);
}