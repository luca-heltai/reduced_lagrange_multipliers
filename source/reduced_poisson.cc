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

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 */

#include "reduced_poisson.h"

#include <boost/algorithm/string.hpp>

#include <type_traits>

#include "augmented_lagrangian_preconditioner.h"



template <int spacedim>
ReducedPoissonParameters<spacedim>::ReducedPoissonParameters()
  : ParameterAcceptor("/Reduced Poisson/")
  , rhs("/Reduced Poisson/Right hand side")
  , bc("/Reduced Poisson/Dirichlet boundary conditions")
  , inner_control("/Reduced Poisson/Solver/Inner control")
  , outer_control("/Reduced Poisson/Solver/Outer control")
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("Output results also before solving",
                output_results_before_solving);
  add_parameter("Solver type", solver_name);
  add_parameter("Initial refinement", initial_refinement);
  add_parameter("Dirichlet boundary ids", dirichlet_ids);
  enter_subsection("Grid generation");
  {
    add_parameter("Grid generator", name_of_grid);
    add_parameter("Grid generator arguments", arguments_for_grid);
  }
  leave_subsection();
  enter_subsection("Refinement and remeshing");
  {
    add_parameter("Strategy",
                  refinement_strategy,
                  "",
                  this->prm,
                  Patterns::Selection("fixed_fraction|fixed_number|global"));
    add_parameter("Coarsening fraction", coarsening_fraction);
    add_parameter("Refinement fraction", refinement_fraction);
    add_parameter("Maximum number of cells", max_cells);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
  }
  leave_subsection();

  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();
}


template <int dim, int spacedim>
ReducedPoisson<dim, spacedim>::ReducedPoisson(
  const ReducedPoissonParameters<spacedim> &par)
  : par(par)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , tria(mpi_communicator,
         typename Triangulation<spacedim>::MeshSmoothing(
           Triangulation<spacedim>::smoothing_on_refinement |
           Triangulation<spacedim>::smoothing_on_coarsening),
         parallel::distributed::Triangulation<
           spacedim>::construct_multigrid_hierarchy)
  , dh(tria)
  , reduced_coupling(tria, par.reduced_coupling_parameters)
  , mapping(1)
{}


template <int dim, int spacedim>
void
read_grid_and_cad_files(const std::string            &grid_file_name,
                        const std::string            &ids_and_cad_file_names,
                        Triangulation<dim, spacedim> &tria)
{
  GridIn<dim, spacedim> grid_in;
  grid_in.attach_triangulation(tria);
  grid_in.read(grid_file_name);
#ifdef DEAL_II_WITH_OPENCASCADE
  using map_type  = std::map<types::manifold_id, std::string>;
  using Converter = Patterns::Tools::Convert<map_type>;
  for (const auto &pair : Converter::to_value(ids_and_cad_file_names))
    {
      const auto &manifold_id   = pair.first;
      const auto &cad_file_name = pair.second;
      const auto  extension     = boost::to_lower_copy(
        cad_file_name.substr(cad_file_name.find_last_of('.') + 1));
      TopoDS_Shape shape;
      if (extension == "iges" || extension == "igs")
        shape = OpenCASCADE::read_IGES(cad_file_name);
      else if (extension == "step" || extension == "stp")
        shape = OpenCASCADE::read_STEP(cad_file_name);
      else
        AssertThrow(false,
                    ExcNotImplemented("We found an extension that we "
                                      "do not recognize as a CAD file "
                                      "extension. Bailing out."));
      const auto n_elements = OpenCASCADE::count_elements(shape);
      if ((std::get<0>(n_elements) == 0))
        tria.set_manifold(
          manifold_id,
          OpenCASCADE::ArclengthProjectionLineManifold<dim, spacedim>(shape));
      else if (spacedim == 3)
        {
          const auto t = reinterpret_cast<Triangulation<dim, 3> *>(&tria);
          t->set_manifold(manifold_id,
                          OpenCASCADE::NormalToMeshProjectionManifold<dim, 3>(
                            shape));
        }
      else
        tria.set_manifold(manifold_id,
                          OpenCASCADE::NURBSPatchManifold<dim, spacedim>(
                            TopoDS::Face(shape)));
    }
#else
  (void)ids_and_cad_file_names;
  AssertThrow(false, ExcNotImplemented("Generation of the grid failed."));
#endif
}



template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::make_grid()
{
  try
    {
      GridGenerator::generate_from_name_and_arguments(tria,
                                                      par.name_of_grid,
                                                      par.arguments_for_grid);
    }
  catch (...)
    {
      pcout << "Generating from name and argument failed." << std::endl
            << "Trying to read from file name." << std::endl;
      read_grid_and_cad_files(par.name_of_grid, par.arguments_for_grid, tria);
    }
  tria.refine_global(par.initial_refinement);
}



template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::setup_fe()
{
  TimerOutput::Scope t(computing_timer, "Initial setup");
  fe = std::make_unique<FESystem<spacedim>>(FE_Q<spacedim>(par.fe_degree), 1);
  quadrature = std::make_unique<QGauss<spacedim>>(par.fe_degree + 1);
}


template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup dofs");
  dh.distribute_dofs(*fe);
#ifdef MATRIX_FREE_PATH
  dh.distribute_mg_dofs();
#endif

  owned_dofs.resize(2);
  owned_dofs[0] = dh.locally_owned_dofs();
  relevant_dofs.resize(2);
  DoFTools::extract_locally_relevant_dofs(dh, relevant_dofs[0]);
  {
    constraints.reinit(owned_dofs[0], relevant_dofs[0]);
    DoFTools::make_hanging_node_constraints(dh, constraints);
    for (const auto id : par.dirichlet_ids)
      VectorTools::interpolate_boundary_values(dh, id, par.bc, constraints);
    constraints.close();
  }
  {
#ifdef MATRIX_FREE_PATH
    typename MatrixFree<spacedim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<spacedim, double>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    std::shared_ptr<MatrixFree<spacedim, double>> system_mf_storage(
      new MatrixFree<spacedim, double>());
    system_mf_storage->reinit(
      mapping, dh, constraints, QGauss<1>(fe->degree + 1), additional_data);
    stiffness_matrix.initialize(system_mf_storage);

    // Perform setup for matrix-free multigrid
    {
      const unsigned int nlevels = tria.n_global_levels();
      mg_matrices.resize(0, nlevels - 1);

      const std::set<types::boundary_id> dirichlet_boundary_ids = {0};
      mg_constrained_dofs.initialize(dh);
      mg_constrained_dofs.make_zero_boundary_constraints(
        dh, dirichlet_boundary_ids);

      for (unsigned int level = 0; level < nlevels; ++level)
        {
          AffineConstraints<double> level_constraints(
            dh.locally_owned_mg_dofs(level),
            DoFTools::extract_locally_relevant_level_dofs(dh, level));
          for (const types::global_dof_index dof_index :
               mg_constrained_dofs.get_boundary_indices(level))
            level_constraints.constrain_dof_to_zero(dof_index);
          level_constraints.close();

          typename MatrixFree<spacedim, float>::AdditionalData
            additional_data_level;
          additional_data_level.tasks_parallel_scheme =
            MatrixFree<spacedim, float>::AdditionalData::none;
          additional_data_level.mapping_update_flags =
            (update_gradients | update_JxW_values | update_quadrature_points);
          additional_data_level.mg_level = level;
          std::shared_ptr<MatrixFree<spacedim, float>> mg_mf_storage_level =
            std::make_shared<MatrixFree<spacedim, float>>();
          mg_mf_storage_level->reinit(mapping,
                                      dh,
                                      level_constraints,
                                      QGauss<1>(fe->degree + 1),
                                      additional_data_level);

          mg_matrices[level].initialize(mg_mf_storage_level,
                                        mg_constrained_dofs,
                                        level);
        }
    }


#else
    stiffness_matrix.clear();
    DynamicSparsityPattern dsp(relevant_dofs[0]);
    DoFTools::make_sparsity_pattern(dh, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs[0],
                                               mpi_communicator,
                                               relevant_dofs[0]);
    stiffness_matrix.reinit(owned_dofs[0],
                            owned_dofs[0],
                            dsp,
                            mpi_communicator);

#endif
  }
  // Initialize the coupling object
  reduced_coupling.initialize(mapping);

  const auto &reduced_dh = reduced_coupling.get_dof_handler();
  owned_dofs[1]          = reduced_dh.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(reduced_dh, relevant_dofs[1]);

  coupling_matrix.clear();

  DynamicSparsityPattern dsp(dh.n_dofs(),
                             reduced_dh.n_dofs(),
                             relevant_dofs[0]);

  reduced_coupling.assemble_coupling_sparsity(dsp, dh, constraints);

  coupling_matrix.reinit(owned_dofs[0], owned_dofs[1], dsp, mpi_communicator);

  //     DynamicSparsityPattern idsp(inclusions.n_dofs(),
  //                                 inclusions.n_dofs(),
  //                                 relevant_dofs[1]);
  //     for (const auto i : relevant_dofs[1])
  //       idsp.add(i, i);
  //     SparsityTools::distribute_sparsity_pattern(idsp,
  //                                                owned_dofs[1],
  //                                                mpi_communicator,
  //                                                relevant_dofs[1]);
  //     inclusion_matrix.reinit(owned_dofs[1],
  //                             owned_dofs[1],
  //                             idsp,
  //                             mpi_communicator);
  //   }

  // Commented out inclusions-dependent reinit
  locally_relevant_solution.reinit(owned_dofs, relevant_dofs, mpi_communicator);

#ifdef MATRIX_FREE_PATH
  system_rhs.reinit(owned_dofs, relevant_dofs, mpi_communicator);
  solution.reinit(owned_dofs, relevant_dofs, mpi_communicator);
#else
  system_rhs.reinit(owned_dofs, mpi_communicator);
  solution.reinit(owned_dofs, mpi_communicator);
#endif

  pcout << "   Number of degrees of freedom: " << owned_dofs[0].size() << " + "
        << owned_dofs[1].size()
        << " (locally owned: " << owned_dofs[0].n_elements() << " + "
        << owned_dofs[1].n_elements() << ")" << std::endl;
}


#ifndef MATRIX_FREE_PATH
template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::assemble_poisson_system()
{
  stiffness_matrix = 0;
  coupling_matrix  = 0;
  system_rhs       = 0;
  FEValues<spacedim>               fe_values(*fe,
                               *quadrature,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  const unsigned int               dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int               n_q_points    = quadrature->size();
  FullMatrix<double>               cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>                   cell_rhs(dofs_per_cell);
  std::vector<double>              rhs_values(n_q_points);
  std::vector<Tensor<1, spacedim>> grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Scalar     scalar(0);
  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        par.rhs.value_list(fe_values.get_quadrature_points(), rhs_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              grad_phi_u[k] = fe_values[scalar].gradient(k, q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      grad_phi_u[i] * grad_phi_u[j] * fe_values.JxW(q);
                  }
                cell_rhs(i) += fe_values.shape_value(i, q) * rhs_values[q] *
                               fe_values.JxW(q);
              }
          }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               stiffness_matrix,
                                               system_rhs.block(0));
      }
  stiffness_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}
#endif

// Commented out inclusions-dependent function
/*
template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::assemble_coupling()
{
  TimerOutput::Scope t(computing_timer, "Assemble Coupling matrix");
  pcout << "Assemble coupling matrix. " << std::endl;

  std::vector<types::global_dof_index>
fe_dof_indices(fe->n_dofs_per_cell()); std::vector<types::global_dof_index>
inclusion_dof_indices( inclusions.get_n_coefficients());

  FullMatrix<double> local_coupling_matrix(fe->n_dofs_per_cell(),
                                           inclusions.get_n_coefficients());

  [[maybe_unused]] FullMatrix<double>
local_bulk_matrix(fe->n_dofs_per_cell(), fe->n_dofs_per_cell());

  FullMatrix<double> local_inclusion_matrix(inclusions.get_n_coefficients(),
                                            inclusions.get_n_coefficients());

  Vector<double> local_rhs(inclusions.get_n_coefficients());

  auto particle = inclusions.inclusions_as_particles.begin();
  while (particle != inclusions.inclusions_as_particles.end())
    {
      const auto &cell = particle->get_surrounding_cell();
      const auto  dh_cell =
        typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
      dh_cell->get_dof_indices(fe_dof_indices);
      const auto pic =
        inclusions.inclusions_as_particles.particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());

      auto p      = pic.begin();
      auto next_p = pic.begin();
      while (p != pic.end())
        {
          const auto inclusion_id =
inclusions.get_inclusion_id(p->get_id()); inclusion_dof_indices   =
inclusions.get_dof_indices(p->get_id()); local_coupling_matrix   = 0;
          local_inclusion_matrix  = 0;
          local_rhs               = 0;

          // Extract all points that refer to the same inclusion
          std::vector<Point<spacedim>> ref_q_points;
          for (; next_p != pic.end() &&
                 inclusions.get_inclusion_id(next_p->get_id()) ==
inclusion_id;
               ++next_p)
            ref_q_points.push_back(next_p->get_reference_location());
          FEValues<spacedim, spacedim> fev(*fe,
                                           ref_q_points,
                                           update_values | update_gradients);
          fev.reinit(dh_cell);
          for (unsigned int q = 0; q < ref_q_points.size(); ++q)
            {
              const auto  id                  = p->get_id();
              const auto &inclusion_fe_values =
inclusions.get_fe_values(id); const auto &real_q              =
p->get_location();

              // Coupling and inclusions matrix
              for (unsigned int j = 0; j < inclusions.get_n_coefficients();
++j)
                {
                  for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
                    local_coupling_matrix(i, j) +=
                      (fev.shape_value(i, q)) * inclusion_fe_values[j];
                  local_rhs(j) +=
                    inclusion_fe_values[j] *
                    inclusions.get_inclusion_data(inclusion_id, id, real_q);

                  local_inclusion_matrix(j, j) +=
                    (inclusion_fe_values[j] * inclusion_fe_values[j] /
                     inclusion_fe_values[0]);
                }
              ++p;
            }
          // I expect p and next_p to be the same now.
          Assert(p == next_p, ExcInternalError());

          // Add local matrices to global ones
          constraints.distribute_local_to_global(local_coupling_matrix,
                                                 fe_dof_indices,
                                                 inclusion_constraints,
                                                 inclusion_dof_indices,
                                                 coupling_matrix);
          inclusion_constraints.distribute_local_to_global(
            local_rhs, inclusion_dof_indices, system_rhs.block(1));

          inclusion_constraints.distribute_local_to_global(
            local_inclusion_matrix, inclusion_dof_indices,
inclusion_matrix);
        }
      particle = pic.end();
    }
  coupling_matrix.compress(VectorOperation::add);
#ifndef MATRIX_FREE_PATH
  stiffness_matrix.compress(VectorOperation::add);
#endif
  inclusion_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  pcout << "System rhs: " << system_rhs.l2_norm() << std::endl;
}
*/

#ifdef MATRIX_FREE_PATH
template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::assemble_rhs()
{
  stiffness_matrix.get_matrix_free()->initialize_dof_vector(solution.block(0));
  stiffness_matrix.get_matrix_free()->initialize_dof_vector(
    system_rhs.block(0));
  system_rhs        = 0;
  solution.block(0) = 0;
  constraints.distribute(solution.block(0));
  solution.block(0).update_ghost_values();

  FEEvaluation<spacedim, -1> phi(*stiffness_matrix.get_matrix_free());
  for (unsigned int cell = 0;
       cell < stiffness_matrix.get_matrix_free()->n_cell_batches();
       ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(solution.block(0));
      phi.evaluate(EvaluationFlags::gradients);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const Point<spacedim, VectorizedArray<double>> p_vect =
            phi.quadrature_point(q);

          VectorizedArray<double> f_value = 0.0;
          for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
            {
              Point<spacedim> p;
              for (unsigned int d = 0; d < spacedim; ++d)
                p[d] = p_vect[d][v];
              f_value[v] = par.rhs.value(p);
            }
          phi.submit_gradient(-phi.get_gradient(q), q);
          phi.submit_value(f_value, q);
        }
      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi.distribute_local_to_global(system_rhs.block(0));
    }

  system_rhs.compress(VectorOperation::add);
}
#endif


template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::solve()
{
  TimerOutput::Scope t(computing_timer, "Solve");
  pcout << "Preparing solve." << std::endl;
  SolverCG<VectorType> cg_stiffness(par.inner_control);
#ifdef MATRIX_FREE_PATH

  using Payload = dealii::internal::LinearOperatorImplementation::EmptyPayload;
  LinearOperator<VectorType, VectorType, Payload> A;
  A = linear_operator<VectorType, VectorType, Payload>(stiffness_matrix);

  MGTransferMatrixFree<spacedim, float> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dh);

  using SmootherType =
    PreconditionChebyshev<LevelMatrixType,
                          LinearAlgebra::distributed::Vector<float>>;
  mg::SmootherRelaxation<SmootherType,
                         LinearAlgebra::distributed::Vector<float>>
                                                       mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, tria.n_global_levels() - 1);
  for (unsigned int level = 0; level < tria.n_global_levels(); ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 15.;
          smoother_data[level].degree              = 5;
          smoother_data[level].eig_cg_n_iterations = 10;
        }
      else
        {
          smoother_data[0].smoothing_range     = 1e-3;
          smoother_data[0].degree              = numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
      mg_matrices[level].compute_diagonal();
      smoother_data[level].preconditioner =
        mg_matrices[level].get_matrix_diagonal_inverse();
    }
  mg_smoother.initialize(mg_matrices, smoother_data);

  MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>>
    mg_coarse;
  mg_coarse.initialize(mg_smoother);

  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
    mg_interface_matrices;
  mg_interface_matrices.resize(0, tria.n_global_levels() - 1);
  for (unsigned int level = 0; level < tria.n_global_levels(); ++level)
    mg_interface_matrices[level].initialize(mg_matrices[level]);
  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(
    mg_interface_matrices);

  Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
    mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  mg.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<spacedim,
                 LinearAlgebra::distributed::Vector<float>,
                 MGTransferMatrixFree<spacedim, float>>
    preconditioner(dh, mg, mg_transfer);

  auto invA = A;
  invA      = inverse_operator(A, cg_stiffness, preconditioner);
#else
  using Payload =
    TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
  LinearOperator<VectorType, VectorType, Payload> A;
  A = linear_operator<VectorType, VectorType, Payload>(stiffness_matrix);

  LA::MPI::PreconditionAMG prec_A;
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
#  ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#  endif
    pcout << "Initialize AMG...";
    prec_A.initialize(stiffness_matrix, data);
    pcout << "done." << std::endl;
  }
  const auto amgA = linear_operator<VectorType, VectorType, Payload>(A, prec_A);
  auto       invA = A;
  invA            = inverse_operator(A, cg_stiffness, amgA);
#endif


  // Some aliases
  auto &u      = solution.block(0);
  auto &lambda = solution.block(1);

  const auto &f = system_rhs.block(0);
  const auto &g = system_rhs.block(1);

  if (reduced_coupling.get_dof_handler().n_dofs() == 0)
    {
      u = invA * f;
    }
  else
    {
#ifdef MATRIX_FREE_PATH
      auto Bt =
        linear_operator<VectorType, VectorType, Payload>(*coupling_operator);
      Bt.reinit_range_vector = [this](VectorType &vec, const bool) {
        vec.reinit(owned_dofs[0], relevant_dofs[0], mpi_communicator);
      };
      Bt.reinit_domain_vector = [this](VectorType &vec, const bool) {
        vec.reinit(owned_dofs[1], relevant_dofs[1], mpi_communicator);
      };

      const auto B = transpose_operator<VectorType, VectorType, Payload>(Bt);
#else
      const auto Bt =
        linear_operator<VectorType, VectorType, Payload>(coupling_matrix);
      const auto B = transpose_operator<VectorType, VectorType, Payload>(Bt);
      // const auto B = linear_operator<VectorType, VectorType, Payload>(
      //   coupling_matrix_transpose);
#endif

      if (par.solver_name == "Schur")
        {
          // Schur complement
          pcout << "   Prepare schur... ";
          const auto S = B * invA * Bt;
          pcout << "S was built." << std::endl;

          // Schur complement preconditioner
          auto                     invS = S;
          SolverFGMRES<VectorType> solver_schur(par.outer_control);
          invS =
            inverse_operator<Payload, SolverFGMRES<VectorType>>(S,
                                                                solver_schur);

          pcout << "   f norm: " << f.l2_norm() << ", g norm: " << g.l2_norm()
                << std::endl;

          // Compute Lambda first
          lambda = invS * (B * invA * f - g);
          pcout << "   Solved for lambda in " << par.outer_control.last_step()
                << " iterations." << std::endl;

          // Then compute u
          u = invA * (f - Bt * lambda);
          pcout << "   u norm: " << u.l2_norm()
                << ", lambda norm: " << lambda.l2_norm() << std::endl;

          pcout << "   Solved for u in " << par.inner_control.last_step()
                << " iterations." << std::endl;

          constraints.distribute(u);
          reduced_coupling.get_coupling_constraints().distribute(lambda);
          // solution.update_ghost_values();
          locally_relevant_solution = solution;
        }
      else if (par.solver_name == "AL")
        {
          pcout << "Prepare AL preconditioner... " << std::endl;

          AssertThrow(
            (std::is_same_v<LA::MPI::Vector, TrilinosWrappers::MPI::Vector>),
            ExcNotImplemented());

          LA::MPI::SparseMatrix reduced_mass_matrix;
          const auto           &reduced_dh = reduced_coupling.get_dof_handler();
          DynamicSparsityPattern dsp_reduced_mass(relevant_dofs[1]);
          DoFTools::make_sparsity_pattern(reduced_dh, dsp_reduced_mass);
          SparsityTools::distribute_sparsity_pattern(dsp_reduced_mass,
                                                     owned_dofs[1],
                                                     mpi_communicator,
                                                     relevant_dofs[1]);
          reduced_mass_matrix.reinit(owned_dofs[1],
                                     owned_dofs[1],
                                     dsp_reduced_mass,
                                     mpi_communicator);

          // Create mass matrix associated with the reduced dof handler
          reduced_coupling.assemble_coupling_mass_matrix(reduced_mass_matrix);

          pcout << "Reduced mass matrix size: " << reduced_mass_matrix.m()
                << " x " << reduced_mass_matrix.n()
                << ", norm: " << reduced_mass_matrix.linfty_norm() << std::endl;


          const auto M = linear_operator<LA::MPI::Vector>(reduced_mass_matrix);

          // Augmented Lagrangian solver
          TrilinosWrappers::PreconditionILU M_inv_ilu;
          M_inv_ilu.initialize(reduced_mass_matrix);

          TrilinosWrappers::MPI::Vector inverse_squares_reduced; // diag(M)^{-2}
          inverse_squares_reduced.reinit(owned_dofs[1], mpi_communicator);
          for (const types::global_dof_index local_idx : owned_dofs[1])
            {
              const double el = reduced_mass_matrix.diag_element(local_idx);
              Assert(std::abs(el) > 1e-10,
                     ExcMessage(
                       "Diagonal element " + std::to_string(local_idx) +
                       " of reduced mass matrix (" + std::to_string(el) +
                       ") is close to zero. Cannot compute inverse square."));
              inverse_squares_reduced(local_idx) = 1. / (el * el);
            }

          inverse_squares_reduced.compress(VectorOperation::insert);


          SolverControl solver_control(100, 1e-15, false, false);
          SolverCG<TrilinosWrappers::MPI::Vector> solver_mass_matrix(
            solver_control);
          auto invM = inverse_operator(M, solver_mass_matrix, M_inv_ilu);
          auto invW = invM * invM;

          const double gamma = 10; // TODO: add to parameters file
          auto         Aug   = A + gamma * Bt * invW * B;

          TrilinosWrappers::SparseMatrix augmented_matrix;
          pcout << "Building augmented matrix..." << std::endl;
          UtilitiesAL::create_augmented_block(stiffness_matrix,
                                              coupling_matrix,
                                              inverse_squares_reduced,
                                              gamma,
                                              augmented_matrix);
          pcout << "done." << std::endl;

          TrilinosWrappers::PreconditionAMG prec_amg_augmented_block;
          TrilinosWrappers::PreconditionAMG::AdditionalData data;
          prec_amg_augmented_block.initialize(augmented_matrix, data);

          auto Zero = M * 0.0;
          auto AA   = block_operator<2, 2, LA::MPI::BlockVector>(
            {{{{Aug, Bt}}, {{B, Zero}}}}); //! Augmented the (1,1) block

          LA::MPI::BlockVector solution_block;
          LA::MPI::BlockVector system_rhs_block;
          AA.reinit_domain_vector(solution_block, false);
          AA.reinit_range_vector(system_rhs_block, false);

          // lagrangian term
          LA::MPI::Vector tmp;
          tmp.reinit(system_rhs.block(0));
          tmp                       = gamma * Bt * invW * system_rhs.block(1);
          system_rhs_block.block(0) = system_rhs.block(0);
          system_rhs_block.block(0).add(1., tmp); // ! augmented
          system_rhs_block.block(1) = system_rhs.block(1);

          SolverCG<LA::MPI::Vector> solver_lagrangian(par.inner_control);


          auto Aug_inv =
            inverse_operator(Aug, solver_lagrangian, prec_amg_augmented_block);
          SolverFGMRES<LA::MPI::BlockVector> solver_fgmres(par.outer_control);

          UtilitiesAL::BlockPreconditionerAugmentedLagrangian<LA::MPI::Vector>
            augmented_lagrangian_preconditioner{Aug_inv, B, Bt, invW, gamma};

          solver_fgmres.solve(AA,
                              solution_block,
                              system_rhs_block,
                              augmented_lagrangian_preconditioner);

          pcout << "   Solved with AL preconditioner in "
                << par.outer_control.last_step() << " iterations." << std::endl;

          constraints.distribute(solution_block.block(0));
          reduced_coupling.get_coupling_constraints().distribute(
            solution_block.block(1));
          // solution.update_ghost_values();
          locally_relevant_solution = solution_block;


#ifdef DEBUG
          // Estimate condition number of BBt using CG
          {
            auto output_double_number = [this](double             input,
                                               const std::string &text) {
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                std::cout << text << input << std::endl;
            };

            // Estimate condition number:
            pcout << "- - - - - - - - - - - - - - - - - - - - - - - -"
                  << std::endl;
            pcout << "Estimate condition number of BBt using CG" << std::endl;
            SolverControl solver_control(100000, 1e-12);
            SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_control);

            solver_cg.connect_condition_number_slot(
              std::bind(output_double_number,
                        std::placeholders::_1,
                        "Condition number estimate: "));
            using PayloadType = dealii::TrilinosWrappers::internal::
              LinearOperatorImplementation::TrilinosPayload;

            auto BBt = B * Bt;

            TrilinosWrappers::MPI::Vector u(lambda);
            u = 0.;
            TrilinosWrappers::MPI::Vector f(lambda);
            f = 1.;
            TrilinosWrappers::PreconditionIdentity prec_no;
            try
              {
                solver_cg.solve(BBt, u, f, prec_no);
              }
            catch (...)
              {
                pcout
                  << "***BBt solve not successfull (see condition number above)***"
                  << std::endl;
              }
          }
#endif
        }
      else
        {
          DEAL_II_NOT_IMPLEMENTED();
        }
    }
}



template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::refine_and_transfer()
{
  TimerOutput::Scope t(computing_timer, "Refine");
  Vector<float>      error_per_cell(tria.n_active_cells());
  KellyErrorEstimator<spacedim>::estimate(dh,
                                          QGauss<spacedim - 1>(par.fe_degree +
                                                               1),
                                          {},
                                          locally_relevant_solution.block(0),
                                          error_per_cell);
  if (par.refinement_strategy == "fixed_fraction")
    {
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        tria, error_per_cell, par.refinement_fraction, par.coarsening_fraction);
    }
  else if (par.refinement_strategy == "fixed_number")
    {
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        error_per_cell,
        par.refinement_fraction,
        par.coarsening_fraction,
        par.max_cells);
    }
  else if (par.refinement_strategy == "global")
    for (const auto &cell : tria.active_cell_iterators())
      cell->set_refine_flag();

  parallel::distributed::SolutionTransfer<spacedim, VectorType> transfer(dh);
  tria.prepare_coarsening_and_refinement();
  // inclusions.inclusions_as_particles.prepare_for_coarsening_and_refinement();
  transfer.prepare_for_coarsening_and_refinement(
    locally_relevant_solution.block(0));
  tria.execute_coarsening_and_refinement();
  // inclusions.inclusions_as_particles.unpack_after_coarsening_and_refinement();
  setup_dofs();
  transfer.interpolate(solution.block(0));
  constraints.distribute(solution.block(0));
  locally_relevant_solution.block(0) = solution.block(0);
}



template <int dim, int spacedim>
std::string
ReducedPoisson<dim, spacedim>::output_solution() const
{
  TimerOutput::Scope t(computing_timer, "Output results");
  std::string        solution_name = "solution";
  DataOut<spacedim>  data_out;
  data_out.attach_dof_handler(dh);
  data_out.add_data_vector(locally_relevant_solution.block(0), solution_name);
  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();
  const std::string filename =
    par.output_name + "_" + std::to_string(cycle) + ".vtu";
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                 mpi_communicator);
  return filename;
}


template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::output_results() const
{
  static std::vector<std::pair<double, std::string>> cycles_and_solutions;
  static std::vector<std::pair<double, std::string>> cycles_and_particles;

  if (cycles_and_solutions.size() == cycle)
    {
      cycles_and_solutions.push_back({(double)cycle, output_solution()});

      const std::string particles_filename =
        par.output_name + "_particles_" + std::to_string(cycle) + ".vtu";
      reduced_coupling.output_particles(par.output_directory + "/" +
                                        particles_filename);

      cycles_and_particles.push_back({(double)cycle, particles_filename});

      std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
                                  ".pvd");
      std::ofstream pvd_particles(par.output_directory + "/" + par.output_name +
                                  "_particles.pvd");
      DataOutBase::write_pvd_record(pvd_solutions, cycles_and_solutions);
      DataOutBase::write_pvd_record(pvd_particles, cycles_and_particles);
    }
}

template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::print_parameters() const
{
#ifdef USE_PETSC_LA
  pcout << "Running ReducedPoisson<" << Utilities::dim_string(dim, spacedim)
        << "> using PETSc." << std::endl;
#else
  pcout << "Running ReducedPoisson<" << Utilities::dim_string(dim, spacedim)
        << "> using Trilinos with "
        << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << " MPI ranks."
        << std::endl;
#endif
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      par.prm.print_parameters(par.output_directory + "/" + "used_parameters_" +
                                 std::to_string(dim) +
                                 std::to_string(spacedim) + ".prm",
                               ParameterHandler::Short);
    }
}

template <int dim, int spacedim>
void
ReducedPoisson<dim, spacedim>::run()
{
  print_parameters();
  make_grid();
  setup_fe();
  for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
    {
      setup_dofs();
      if (par.output_results_before_solving)
        output_results();
#ifdef MATRIX_FREE_PATH
      assemble_rhs();
#else
      assemble_poisson_system();
#endif
      reduced_coupling.assemble_coupling_matrix(coupling_matrix,
                                                dh,
                                                constraints);
      reduced_coupling.assemble_reduced_rhs(system_rhs.block(1));

#ifdef MATRIX_FREE_PATH
      // MappingQ1<spacedim> mapping;
      // coupling_operator =
      // std::make_unique<CouplingOperator<spacedim,
      // double>>(
      //   inclusions, dh, constraints,
      //   mapping, *fe);
#endif
      // return;
      solve();
      output_results();
      par.convergence_table.error_from_exact(dh,
                                             locally_relevant_solution.block(0),
                                             par.bc);
      if (cycle != par.n_refinement_cycles - 1)
        refine_and_transfer();
      if (pcout.is_active())
        par.convergence_table.output_table(pcout.get_stream());
    }
}


// Template instantiations
template class ReducedPoissonParameters<2>;
template class ReducedPoissonParameters<3>;

template class ReducedPoisson<2>;
template class ReducedPoisson<3>;
