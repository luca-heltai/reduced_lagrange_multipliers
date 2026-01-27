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

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <fstream>
#include <optional>
#include <vector>

#include "utils.h"

using namespace dealii;

TEST(FESpaceOnNetwork, VertexDoFsContinuousAtJunctions) // NOLINT
{
  static constexpr int dim      = 1;
  static constexpr int spacedim = 3;

  Triangulation<dim, spacedim> network_tria;
  GridIn<dim, spacedim>        grid_in;
  grid_in.attach_triangulation(network_tria);

  std::ifstream input_file(SOURCE_DIR "/data/tests/mstree_100.vtk");
  ASSERT_TRUE(input_file.good());
  grid_in.read_vtk(input_file);

  DoFHandler<dim, spacedim> dof_handler(network_tria);
  FE_Q<dim, spacedim>       fe(1);
  dof_handler.distribute_dofs(fe);

  const auto junction_faces = get_non_manifold_faces(network_tria);
  ASSERT_GT(junction_faces.size(), 0u);

  for (const auto &[face, cells] : junction_faces)
    {
      ASSERT_GE(cells.size(), 3u);

      std::optional<types::global_dof_index> reference_dof;

      for (const auto &cell : cells)
        {
          const bool         vertex0_at_junction = (cell->face(0) == face);
          const unsigned int junction_vertex_idx = vertex0_at_junction ? 0 : 1;

          typename DoFHandler<dim, spacedim>::active_cell_iterator dof_cell(
            &network_tria, cell->level(), cell->index(), &dof_handler);

          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          dof_cell->get_dof_indices(dof_indices);

          ASSERT_GT(fe.dofs_per_vertex, 0u);
          const auto dof_idx =
            dof_indices[fe.dofs_per_vertex * junction_vertex_idx];

          if (!reference_dof)
            reference_dof = dof_idx;

          EXPECT_EQ(dof_idx, *reference_dof)
            << "Mismatch in vertex DoF at a junction.";
        }
    }
}

