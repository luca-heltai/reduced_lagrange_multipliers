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

#ifndef utils_h
#define utils_h

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>


using namespace dealii;

struct RefinementParameters
{
  RefinementParameters()
  {
    use_space                       = false;
    use_embedded                    = false;
    apply_delta_refinements         = false;
    space_pre_refinement_cycles     = 1;
    embedded_post_refinement_cycles = 1;
  }

  bool         use_space                       = false;
  bool         use_embedded                    = false;
  bool         apply_delta_refinements         = false;
  unsigned int space_pre_refinement_cycles     = 1;
  int          embedded_post_refinement_cycles = 0;
};


namespace GridUtils
{
  template <int reduced_dim, int spacedim>
  void
  adjust_grids(Triangulation<spacedim, spacedim>    &space_triangulation,
               Triangulation<reduced_dim, spacedim> &embedded_triangulation,
               const RefinementParameters &parameters = RefinementParameters())
  {
    Assert(
      (dynamic_cast<parallel::TriangulationBase<reduced_dim, spacedim> *>(
         &embedded_triangulation) == nullptr),
      ExcMessage(
        "The embedded triangulation must not be distributed. It will be partitioned later."));

    namespace bgi = boost::geometry::index;

    // build caches so that we can get local trees
    GridTools::Cache<spacedim, spacedim>    space_cache{space_triangulation};
    GridTools::Cache<reduced_dim, spacedim> embedded_cache{
      embedded_triangulation};

    auto refine = [&]() {
      bool done        = false;
      bool global_done = false;

      double min_embedded = 1e10;
      double max_embedded = 0;
      double min_space    = 1e10;
      double max_space    = 0;

      while (global_done == false)
        {
          done = true;

          // Bounding boxes of the space grid
          const auto &tree =
            space_cache.get_locally_owned_cell_bounding_boxes_rtree();

          // Bounding boxes of the embedded grid
          const auto &embedded_tree =
            embedded_cache.get_cell_bounding_boxes_rtree();

          // Let's check all cells whose bounding box contains an embedded
          // bounding box
          const bool use_space    = parameters.use_space;
          const bool use_embedded = parameters.use_embedded;

          AssertThrow(!(use_embedded && use_space),
                      ExcMessage("You can't refine both the embedded and "
                                 "the space grid at the same time."));

          for (const auto &[embedded_box, embedded_cell] : embedded_tree)
            {
              const auto &[p1, p2] = embedded_box.get_boundary_points();
              const auto diameter  = p1.distance(p2);
              min_embedded         = std::min(min_embedded, diameter);
              max_embedded         = std::max(max_embedded, diameter);

              for (const auto &[space_box, space_cell] :
                   tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
                {
                  const auto &[sp1, sp2]    = space_box.get_boundary_points();
                  const auto space_diameter = sp1.distance(sp2);
                  min_space = std::min(min_space, space_diameter);
                  max_space = std::max(max_space, space_diameter);

                  if (use_embedded && space_diameter < diameter)
                    {
                      embedded_cell->set_refine_flag();
                      done = false;
                    }
                  if (use_space && diameter < space_diameter)
                    {
                      space_cell->set_refine_flag();
                      done = false;
                    }
                }
            }

          // Synchronize done variable across all processes, otherwise we might
          // deadlock
          global_done =
            Utilities::MPI::min(static_cast<int>(done),
                                space_triangulation.get_communicator());

          if (global_done == false)
            {
              if (use_embedded)
                {
                  // Compute again the embedded displacement grid
                  embedded_triangulation.execute_coarsening_and_refinement();
                }
              if (use_space)
                {
                  // Compute again the embedded displacement grid
                  space_triangulation.execute_coarsening_and_refinement();
                }
            }
        }
      return std::make_tuple(min_space, max_space, min_embedded, max_embedded);
    };

    // Do the refinement loop once, to make sure we satisfy our criterions
    refine();


    // Pre refine the space grid according to the delta refinement
    if (parameters.apply_delta_refinements &&
        parameters.space_pre_refinement_cycles != 0)
      for (unsigned int i = 0; i < parameters.space_pre_refinement_cycles; ++i)
        {
          const auto &tree =
            space_cache.get_locally_owned_cell_bounding_boxes_rtree();

          const auto &embedded_tree =
            embedded_cache.get_cell_bounding_boxes_rtree();

          for (const auto &[embedded_box, embedded_cell] : embedded_tree)
            for (const auto &[space_box, space_cell] :
                 tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
              space_cell->set_refine_flag();
          space_triangulation.execute_coarsening_and_refinement();

          // Make sure again we satisfy our criterion after the space
          // refinement
          refine();
        }

    // Post refinement on embedded grid is easy
    if (parameters.apply_delta_refinements &&
        parameters.embedded_post_refinement_cycles != 0)
      {
        embedded_triangulation.refine_global(
          parameters.embedded_post_refinement_cycles);
      }

    // Check once again we satisfy our criterion, and record min/max
    const auto [sm, sM, em, eM] = refine();


    if (Utilities::MPI::this_mpi_process(
          space_triangulation.get_communicator()) == 0)
      std::cout << "Space local min/max diameters   : " << sm << "/" << sM
                << std::endl
                << "Embedded space min/max diameters: " << em << "/" << eM
                << std::endl;
  }


  /**
   * Find the vertices of the Y-junctions in a 1D network embedded in 3D. This
   * function is supposed to be used in the context of a triangulation that is
   * not distributed yet.
   */
  template <typename CellContainer>
  std::map<typename CellContainer::active_face_iterator,
           std::vector<typename CellContainer::active_cell_iterator>>
  get_non_manifold_faces(const CellContainer &cell_container)
  {
    // Loop over all cells, and for each cell loop over all faces. Store in
    // the container for each face the list of insisting cells. Then remove
    // from the container all entries that have only two neighbors.

    std::map<typename CellContainer::active_face_iterator,
             std::vector<typename CellContainer::active_cell_iterator>>
      face_to_cells;

    for (const auto &cell : cell_container.active_cell_iterators())
      for (const auto f : cell->face_indices())
        face_to_cells[cell->face(f)].push_back(cell);


    for (auto it = face_to_cells.begin(); it != face_to_cells.end();)
      {
        if (it->second.size() <= 2)
          it = face_to_cells.erase(it);
        else
          ++it;
      }

    return face_to_cells;
  }

  // Helper structure for junction data
  struct JunctionInfo
  {
    Point<3>                             point;
    std::vector<types::global_dof_index> dof_indices;
    std::vector<double>                  dof_values;

    // [TODO] @fdrmrc: use deal.II serialization instead of this.
    // We need this to send data around.
    // For every junction, we pack the following data:
    // 1. The coordinates of the junction
    // 2. The number of dof_indices
    // 3. The dof_indices
    // 4. The dof_values
    void
    pack(std::vector<char> &buffer) const
    {
      // Pack point coordinates
      const double      coords[3]   = {point[0], point[1], point[2]};
      const std::size_t coords_size = 3 * sizeof(double); // 24 bytes
      const char       *coords_data = reinterpret_cast<const char *>(coords);
      buffer.insert(buffer.end(), coords_data, coords_data + coords_size);

      // Pack dof_indices
      const std::size_t n_dofs      = dof_indices.size();
      const char       *n_dofs_data = reinterpret_cast<const char *>(&n_dofs);
      buffer.insert(buffer.end(), n_dofs_data, n_dofs_data + sizeof(n_dofs));

      const char *dof_indices_data =
        reinterpret_cast<const char *>(dof_indices.data());
      buffer.insert(buffer.end(),
                    dof_indices_data,
                    dof_indices_data +
                      n_dofs * sizeof(types::global_dof_index));

      // Pack dof_values
      const char *dof_values_data =
        reinterpret_cast<const char *>(dof_values.data());
      buffer.insert(buffer.end(),
                    dof_values_data,
                    dof_values_data + n_dofs * sizeof(double));
    }

    //
    void
    unpack(const std::vector<char> &buffer, std::size_t &pos)
    {
      // Unpack point coordinates
      double coords[3];
      std::memcpy(coords, buffer.data() + pos, sizeof(coords));
      pos += sizeof(coords);
      point = Point<3>(coords[0], coords[1], coords[2]);

      // Unpack dof_indices
      std::size_t n_dofs;
      std::memcpy(&n_dofs, buffer.data() + pos, sizeof(n_dofs));
      pos += sizeof(n_dofs);

      dof_indices.resize(n_dofs);
      std::memcpy(dof_indices.data(),
                  buffer.data() + pos,
                  n_dofs * sizeof(types::global_dof_index));
      pos += n_dofs * sizeof(types::global_dof_index);

      // Unpack dof_values
      dof_values.resize(n_dofs);
      std::memcpy(dof_values.data(),
                  buffer.data() + pos,
                  n_dofs * sizeof(double));
      pos += n_dofs * sizeof(double);
    }
  };


  // Structure to store DoF usage at junctions
  struct JunctionDoFData
  {
    Point<3>                                        point;
    std::map<types::global_dof_index, unsigned int> dof_usage;

    // For MPI serialization
    void
    pack(std::vector<char> &buffer) const
    {
      // Pack point coordinates
      const double coords[3] = {point[0], point[1], point[2]};
      buffer.insert(buffer.end(),
                    reinterpret_cast<const char *>(coords),
                    reinterpret_cast<const char *>(coords) + sizeof(coords));

      // Pack dof_usage map
      const std::size_t n_dofs = dof_usage.size();
      buffer.insert(buffer.end(),
                    reinterpret_cast<const char *>(&n_dofs),
                    reinterpret_cast<const char *>(&n_dofs) + sizeof(n_dofs));

      for (const auto &[dof_idx, count] : dof_usage)
        {
          buffer.insert(buffer.end(),
                        reinterpret_cast<const char *>(&dof_idx),
                        reinterpret_cast<const char *>(&dof_idx) +
                          sizeof(dof_idx));
          buffer.insert(buffer.end(),
                        reinterpret_cast<const char *>(&count),
                        reinterpret_cast<const char *>(&count) + sizeof(count));
        }
    }

    void
    unpack(const std::vector<char> &buffer, std::size_t &pos)
    {
      // Unpack point coordinates
      double coords[3];
      std::memcpy(coords, buffer.data() + pos, sizeof(coords));
      pos += sizeof(coords);
      point = Point<3>(coords[0], coords[1], coords[2]);

      // Unpack dof_usage map
      std::size_t n_dofs;
      std::memcpy(&n_dofs, buffer.data() + pos, sizeof(n_dofs));
      pos += sizeof(n_dofs);

      for (std::size_t i = 0; i < n_dofs; ++i)
        {
          types::global_dof_index dof_idx;
          unsigned int            count;

          std::memcpy(&dof_idx, buffer.data() + pos, sizeof(dof_idx));
          pos += sizeof(dof_idx);

          std::memcpy(&count, buffer.data() + pos, sizeof(count));
          pos += sizeof(count);

          dof_usage[dof_idx] = count;
        }
    }
  };



} // namespace GridUtils

// Backward-compatible wrapper (other code calls `adjust_grids(...)` directly).
template <int reduced_dim, int spacedim>
inline void
adjust_grids(Triangulation<spacedim, spacedim>    &space_triangulation,
             Triangulation<reduced_dim, spacedim> &embedded_triangulation,
             const RefinementParameters &parameters = RefinementParameters())
{
  GridUtils::adjust_grids(space_triangulation, embedded_triangulation, parameters);
}

#endif
