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

#include "particle_coupling.h"
template <int dim>
ParticleCouplingParameters<dim>::ParticleCouplingParameters()
  : ParameterAcceptor("Particle Coupling")
{
  add_parameter("RTree extraction level", rtree_extraction_level);
}

template <int dim>
ParticleCoupling<dim>::ParticleCoupling(
  const ParticleCouplingParameters<dim> &par)
  : par(par)
  , mpi_communicator(MPI_COMM_WORLD)
{}



template <int dim>
void
ParticleCoupling<dim>::output_particles(const std::string &output_name) const
{
  Particles::DataOut<dim> particles_out;
  particles_out.build_patches(particles);
  particles_out.write_vtu_in_parallel(output_name, mpi_communicator);
}



template <int dim>
void
ParticleCoupling<dim>::initialize_particle_handler(
  const parallel::TriangulationBase<dim> &tria,
  const Mapping<dim>                     &mapp)
{
  tria_background = &tria;
  mapping         = &mapp;
  particles.initialize(*tria_background, *mapping);
  mpi_communicator = tria_background->get_communicator();

  {
    std::vector<BoundingBox<dim>> all_boxes;
    all_boxes.reserve(tria.n_locally_owned_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        all_boxes.emplace_back(cell->bounding_box());
    const auto tree = pack_rtree(all_boxes);
    const auto local_boxes =
      extract_rtree_level(tree, par.rtree_extraction_level);

    global_bounding_boxes =
      Utilities::MPI::all_gather(mpi_communicator, local_boxes);
  }

  Assert(!global_bounding_boxes.empty(),
         ExcInternalError(
           "I was expecting the "
           "global_bounding_boxes to be filled at this stage. "
           "Make sure you fill this vector before trying to use it "
           "here. Bailing out."));
}



template <int dim>
std::vector<std::vector<BoundingBox<dim>>>
ParticleCoupling<dim>::get_global_bounding_boxes() const
{
  return global_bounding_boxes;
};



template <int dim>
const Particles::ParticleHandler<dim> &
ParticleCoupling<dim>::get_particles() const
{
  return particles;
}



template <int dim>
void
ParticleCoupling<dim>::insert_points(const std::vector<Point<dim>> &points)
{
  AssertThrow(tria_background, ExcNotInitialized());
  local_indices_map =
    particles.insert_global_particles(points, global_bounding_boxes);

  // Store given points in particles_positions
  particles_positions.reserve(32);
  for (const Point<dim> &p : points)
    particles_positions.push_back(p);
}


template <int dim>
void
ParticleCoupling<dim>::perform_local_refinement_around_particles()
{
  AssertThrow(tria_background, ExcNotInitialized());
  AssertThrow(particles_positions.size() > 0, ExcNotInitialized());

  auto particle = particles.begin();
  while (particle != particles.end())
    {
      const auto &cell = particle->get_surrounding_cell();
      if (cell->is_locally_owned())
        {
          cell->set_refine_flag();
          for (const auto face_no : cell->face_indices())
            if (!cell->at_boundary(face_no))
              cell->neighbor(face_no)->set_refine_flag();
        }
      ++particle;
    }

  const_cast<parallel::TriangulationBase<dim> *>(&*tria_background)
    ->execute_coarsening_and_refinement();

  // Re-initialize the ParticleHandler with the newly refined background
  // triangulation
  particles.initialize(*tria_background, *mapping);
  {
    std::vector<BoundingBox<dim>> all_boxes;
    all_boxes.reserve(tria_background->n_locally_owned_active_cells());
    for (const auto &cell : tria_background->active_cell_iterators())
      if (cell->is_locally_owned())
        all_boxes.emplace_back(cell->bounding_box());
    const auto tree = pack_rtree(all_boxes);
    const auto local_boxes =
      extract_rtree_level(tree, par.rtree_extraction_level);

    global_bounding_boxes =
      Utilities::MPI::all_gather(mpi_communicator, local_boxes);
  }
  particles.insert_global_particles(particles_positions, global_bounding_boxes);
}

// Explicit instantiations

template class ParticleCouplingParameters<2>;
template class ParticleCouplingParameters<3>;

template class ParticleCoupling<2>;
template class ParticleCoupling<3>;
