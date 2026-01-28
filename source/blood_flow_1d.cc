// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by Luca Heltai
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

#include "blood_flow_1d.h"

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/fe_field_function.h>

#include <algorithm>
#include <cmath>

// BloodFlowParameters implementation
// Python equivalent: modelBloodFlow.py::model.__init__ (line 33)
BloodFlowParameters::BloodFlowParameters()
  : ParameterAcceptor("Blood Flow Parameters")
{
  add_parameter("rho", rho, "Blood density [g/cm³]");
  add_parameter("mu", mu, "Blood dynamic viscosity [g/(cm·s)]");
  add_parameter("vel_profile_coeff",
                vel_profile_coeff,
                "Velocity profile coefficient");
  add_parameter("final_time", final_time, "Final simulation time [s]");
  add_parameter("cfl_number", cfl_number, "CFL number for time stepping");
  add_parameter("output_frequency", output_frequency, "Output frequency");
  add_parameter("mesh_filename", mesh_filename, "Input VTK mesh file");
  add_parameter("fe_degree", fe_degree, "Finite element degree");
  add_parameter("output_directory", output_directory, "Output directory");
  add_parameter("output_basename", output_basename, "Output basename");
  add_parameter("flux_type", flux_type, "Numerical flux type");
  add_parameter("limiter_type", limiter_type, "Slope limiter type");
  add_parameter("reference_pressure", reference_pressure, "Reference pressure");

  // Constitutive law parameters
  add_parameter("tube_law_m", tube_law_m, "Constitutive law exponent m");
  add_parameter("tube_law_n", tube_law_n, "Constitutive law exponent n");

  // Inflow parameters
  add_parameter("inlet_flow_amplitude",
                inlet_flow_amplitude,
                "Inlet flow amplitude scale factor");
  add_parameter("cardiac_cycle_period",
                cardiac_cycle_period,
                "Cardiac cycle period [s]");

  // Default vessel parameters (used when data is missing from mesh)
  add_parameter("default_radius", default_radius, "Default vessel radius [cm]");
  add_parameter("default_wave_speed",
                default_wave_speed,
                "Default wave speed [cm/s]");
  add_parameter("default_wall_thickness",
                default_wall_thickness,
                "Default wall thickness [cm]");
}

// BloodFlow1D implementation
template <int spacedim>
BloodFlow1D<spacedim>::BloodFlow1D(BloodFlowParameters &parameters_)
  : mpi_communicator(MPI_COMM_WORLD)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , pcout(std::cout, (this_mpi_process == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::never,
                    TimerOutput::wall_times)
  , parameters(parameters_)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , fe(FE_DGQ<1, spacedim>(parameters.fe_degree), 2) // 2 components: A and Q
  , current_time(0.0)
  , time_step(0.0)
  , timestep_number(0)
  , output_number(0)
{}

template <int spacedim>
// Python equivalent: main_bfe_network.py::main time loop (lines 234-327)
void
BloodFlow1D<spacedim>::run()
{
  pcout << "Running 1D Blood Flow simulation..." << std::endl;
  pcout << "  MPI processes: " << n_mpi_processes << std::endl;

  {
    TimerOutput::Scope t(computing_timer, "mesh reading and setup");
    read_mesh_and_data();
    setup_system();
    setup_boundary_conditions();

    // Use new initial condition function
    // Python equivalent: vessel.setInitialCondition(pIni, uIni)
    const double initial_pressure = parameters.reference_pressure;
    const double initial_velocity = 10.0; // cm/s - typical resting velocity
    set_initial_conditions(initial_pressure, initial_velocity);
  }

  output_results();

  while (current_time < parameters.final_time)
    {
      compute_time_step();

      if (current_time + time_step > parameters.final_time)
        time_step = parameters.final_time - current_time;

      pcout << "Time step " << timestep_number << " at t=" << current_time
            << " with dt=" << time_step << std::endl;

      {
        TimerOutput::Scope t(computing_timer, "time step update");
        update_solution_time_step();
      }

      if (timestep_number % parameters.output_frequency == 0)
        output_results();
    }

  computing_timer.print_summary();
  pcout << "Simulation completed." << std::endl;
}

template <int spacedim>
// Python equivalent: scripts/convert_net_to_vtk.py::read_net_file (line 19)
void
BloodFlow1D<spacedim>::read_mesh_and_data()
{
  pcout << "Reading mesh from: " << parameters.mesh_filename << std::endl;

  // Read the VTK mesh using VTKUtils
  if constexpr (spacedim == 1)
    {
      Triangulation<1, 1> serial_triangulation;
      VTKUtils::read_vtk(parameters.mesh_filename, serial_triangulation);

      // Convert to distributed triangulation
      auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(serial_triangulation,
                                              mpi_communicator);
      triangulation.create_triangulation(construction_data);
    }
  else
    {
      Triangulation<1, spacedim> serial_triangulation;
      VTKUtils::read_vtk(parameters.mesh_filename, serial_triangulation);

      auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(serial_triangulation,
                                              mpi_communicator);
      triangulation.create_triangulation(construction_data);
    }

  pcout << "  Number of cells: " << triangulation.n_global_active_cells()
        << std::endl;

  // Read vessel data from VTK cell data
  // Expected cell data arrays: vessel_id, length, inlet_radius, outlet_radius,
  // etc.
  Vector<double> vessel_ids, lengths, inlet_radii, outlet_radii, wave_speeds;
  Vector<double> inlet_bc_types, outlet_bc_types;
  Vector<double> resistances1, resistances2, compliances;

  try
    {
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "vessel_id",
                               vessel_ids);
      VTKUtils::read_cell_data(parameters.mesh_filename, "length", lengths);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "inlet_radius",
                               inlet_radii);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "outlet_radius",
                               outlet_radii);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "wave_speed",
                               wave_speeds);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "inlet_bc_type",
                               inlet_bc_types);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "outlet_bc_type",
                               outlet_bc_types);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "resistance1",
                               resistances1);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "resistance2",
                               resistances2);
      VTKUtils::read_cell_data(parameters.mesh_filename,
                               "compliance",
                               compliances);
    }
  catch (const std::exception &e)
    {
      pcout
        << "Warning: Could not read all vessel data from VTK file. Using default values."
        << std::endl;
      pcout << "Error: " << e.what() << std::endl;

      // Use default values for missing data
      const unsigned int n_cells = triangulation.n_global_active_cells();
      vessel_ids.reinit(n_cells);
      lengths.reinit(n_cells);
      inlet_radii.reinit(n_cells);
      outlet_radii.reinit(n_cells);
      wave_speeds.reinit(n_cells);
      inlet_bc_types.reinit(n_cells);
      outlet_bc_types.reinit(n_cells);
      resistances1.reinit(n_cells);
      resistances2.reinit(n_cells);
      compliances.reinit(n_cells);

      // Fill with default values
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int i = cell->global_active_cell_index();
            vessel_ids[i]        = i;
            lengths[i]     = cell->diameter(); // Use cell diameter as length
            inlet_radii[i] = outlet_radii[i] = parameters.default_radius;
            wave_speeds[i]                   = parameters.default_wave_speed;
            inlet_bc_types[i] = outlet_bc_types[i] = 0;
            resistances1[i] = resistances2[i] = 1000.0;
            compliances[i]                    = 1e-6;
          }
    }

  // Create vessel data structures
  vessel_data.resize(triangulation.n_global_active_cells());

  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const auto  cell_index = cell->global_active_cell_index();
          VesselData &vessel     = vessel_data[cell_index];

          vessel.inlet_node_id  = 0; // Will be determined from mesh topology
          vessel.outlet_node_id = 1;
          vessel.length         = lengths[cell_index];
          vessel.inlet_radius   = inlet_radii[cell_index];
          vessel.outlet_radius  = outlet_radii[cell_index];
          vessel.wave_speed     = wave_speeds[cell_index];
          vessel.inlet_bc_type =
            static_cast<unsigned int>(inlet_bc_types[cell_index]);
          vessel.outlet_bc_type =
            static_cast<unsigned int>(outlet_bc_types[cell_index]);
          vessel.resistance1 = resistances1[cell_index];
          vessel.resistance2 = resistances2[cell_index];
          vessel.compliance  = compliances[cell_index];

          // Compute derived parameters
          vessel.reference_area = M_PI * std::pow(vessel.inlet_radius, 2);
          vessel.elastic_modulus =
            parameters.rho * std::pow(vessel.wave_speed, 2);
          vessel.wall_thickness =
            vessel.inlet_radius * 0.1; // Assume 10% of radius

          cell_to_vessel_map[cell->global_active_cell_index()] = cell_index;
        }
    }

  pcout << "  Vessel data loaded for " << vessel_data.size() << " vessels."
        << std::endl;
}

template <int spacedim>
// Python equivalent: main_bfe_network.py::vessel setup (lines 129-135)
void
BloodFlow1D<spacedim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  old_solution.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  pressure.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  velocity.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  pcout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;
  pcout << "  Number of locally owned dofs: " << locally_owned_dofs.size()
        << std::endl;

  // Initialize solution with equilibrium state
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const unsigned int vessel_id =
            cell_to_vessel_map[cell->global_active_cell_index()];
          const VesselData &vessel = vessel_data[vessel_id];

          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              const unsigned int component =
                fe.system_to_component_index(i).first;
              const types::global_dof_index global_dof = cell->dof_index(i);

              if (locally_owned_dofs.is_element(global_dof))
                {
                  if (component == 0) // Area component
                    {
                      solution[global_dof] = vessel.reference_area;
                    }
                  else // Flow rate component
                    {
                      solution[global_dof] = 0.0; // Initially at rest
                    }
                }
            }
        }
    }

  solution.compress(VectorOperation::insert);
  old_solution = solution;

  // Create face-to-cells connectivity map for DG integration
  create_face_connectivity_map();
}

template <int spacedim>
// Python equivalent: boundary condition setup in vessel.py (lines 35-40)
void
BloodFlow1D<spacedim>::setup_boundary_conditions()
{
  // Boundary conditions will be applied during assembly
  pcout << "  Boundary conditions set up." << std::endl;
}

template <int spacedim>
// Python equivalent: numerics.py::computeTimeStep (line 78)
void
BloodFlow1D<spacedim>::compute_time_step()
{
  double max_wave_speed = 0.0;
  double min_cell_size  = std::numeric_limits<double>::max();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const unsigned int vessel_id =
            cell_to_vessel_map[cell->global_active_cell_index()];
          const VesselData &vessel = vessel_data[vessel_id];

          min_cell_size = std::min(min_cell_size, cell->diameter());

          // Estimate maximum wave speed in this cell
          Vector<double> state(2);
          state[0] = vessel.reference_area; // Use reference area for estimate
          state[1] = 0.0;

          const double local_wave_speed =
            compute_wave_speed_at_state(state, vessel);
          max_wave_speed = std::max(max_wave_speed, local_wave_speed);
        }
    }

  // Collect maximum values across all processes
  max_wave_speed = Utilities::MPI::max(max_wave_speed, mpi_communicator);
  min_cell_size  = Utilities::MPI::min(min_cell_size, mpi_communicator);

  time_step = parameters.cfl_number * min_cell_size / max_wave_speed;
}

template <int spacedim>
// Python equivalent: numerics.py::computeNewSolution assembly phase (lines
// 115-200)
void
BloodFlow1D<spacedim>::assemble_system()
{
  system_rhs = 0;

  const QGauss<1> quadrature_formula(fe.degree + 1);
  const QGauss<0> face_quadrature_formula(fe.degree + 1);

  FEValues<1, spacedim> fe_values(fe,
                                  quadrature_formula,
                                  update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);

  FEFaceValues<1, spacedim> fe_face_values(fe,
                                           face_quadrature_formula,
                                           update_values |
                                             update_normal_vectors |
                                             update_quadrature_points |
                                             update_JxW_values);

  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  Vector<double>                       cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Get solution values at quadrature points
  std::vector<Vector<double>> solution_values(n_q_points, Vector<double>(2));
  std::vector<Vector<double>> old_solution_values(n_q_points,
                                                  Vector<double>(2));

  // Face solution values
  std::vector<Vector<double>> solution_values_face(n_face_q_points,
                                                   Vector<double>(2));

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;
          fe_values.reinit(cell);

          const unsigned int vessel_id =
            cell_to_vessel_map[cell->global_active_cell_index()];
          const VesselData &vessel = vessel_data[vessel_id];

          fe_values.get_function_values(solution, solution_values);
          fe_values.get_function_values(old_solution, old_solution_values);

          // Volume integral (time derivative + source terms)
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Vector<double> &U     = solution_values[q];
              const Vector<double> &U_old = old_solution_values[q];

              // Compute source terms (friction)
              Vector<double> source(2);
              source[0] = 0.0; // Continuity equation has no source

              // Momentum equation friction term
              if (U[0] > 1e-12) // Avoid division by zero
                {
                  source[1] = -parameters.vel_profile_coeff * M_PI *
                              parameters.mu / parameters.rho * U[1] / U[0];
                }
              else
                {
                  source[1] = 0.0;
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;

                  // Time derivative term (explicit time stepping)
                  cell_rhs[i] +=
                    ((U[component_i] - U_old[component_i]) / time_step) *
                    fe_values.shape_value(i, q) * fe_values.JxW(q);

                  // Source term
                  cell_rhs[i] -= source[component_i] *
                                 fe_values.shape_value(i, q) * fe_values.JxW(q);
                }
            }

          // Face integrals (numerical fluxes) - Transform to proper deal.II
          // integration
          for (const auto face_number : cell->face_indices())
            {
              fe_face_values.reinit(cell, face_number);
              fe_face_values.get_function_values(solution,
                                                 solution_values_face);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  const Vector<double>      &U_face = solution_values_face[q];
                  const Tensor<1, spacedim> &normal =
                    fe_face_values.normal_vector(q);

                  // Get states for numerical flux computation
                  Vector<double> state_left = U_face;
                  Vector<double> state_right =
                    U_face; // For interior faces, would need neighbor

                  // For boundary faces, apply boundary conditions
                  if (cell->face(face_number)->at_boundary())
                    {
                      // Apply boundary conditions to get right state
                      if (face_number == 0) // Left boundary (inlet)
                        {
                          // Apply inlet boundary condition
                          const auto bc_state =
                            apply_inlet_bc(state_left, vessel, current_time);
                          state_right = bc_state;
                        }
                      else // Right boundary (outlet)
                        {
                          // Apply outlet boundary condition
                          const auto bc_state =
                            apply_outlet_bc(state_left, vessel, current_time);
                          state_right = bc_state;
                        }
                    }
                  else
                    {
                      // For interior faces, get neighbor state using
                      // connectivity map
                      const auto  face            = cell->face(face_number);
                      const auto &connected_cells = face_to_cells_map[face];

                      // Find the neighbor cell (the other cell that shares this
                      // face)
                      typename DoFHandler<1, spacedim>::active_cell_iterator
                           neighbor_cell;
                      bool neighbor_found = false;
                      for (const auto &connected_cell : connected_cells)
                        {
                          if (connected_cell->global_active_cell_index() !=
                              cell->global_active_cell_index())
                            {
                              neighbor_cell  = connected_cell;
                              neighbor_found = true;
                              break;
                            }
                        }

                      if (neighbor_found)
                        {
                          // Get neighbor's state at the quadrature point
                          // For simplicity, use cell average (could be improved
                          // with face interpolation)
                          Vector<double> neighbor_state(2);
                          for (unsigned int comp = 0; comp < 2; ++comp)
                            {
                              // Simple cell average - could be improved
                              neighbor_state[comp]           = 0.0;
                              unsigned int dofs_in_component = 0;
                              for (unsigned int i = 0; i < fe.dofs_per_cell;
                                   ++i)
                                {
                                  if (fe.system_to_component_index(i).first ==
                                      comp)
                                    {
                                      neighbor_state[comp] +=
                                        solution[neighbor_cell->dof_index(i)];
                                      ++dofs_in_component;
                                    }
                                }
                              if (dofs_in_component > 0)
                                neighbor_state[comp] /= dofs_in_component;
                            }
                          state_right = neighbor_state;
                        }
                      else
                        {
                          // Fallback if no neighbor found (shouldn't happen for
                          // well-formed mesh)
                          state_right = state_left;
                        }
                    }

                  // Compute numerical flux using deal.II integration
                  const Vector<double> numerical_flux = compute_hll_flux_vector(
                    state_left, state_right, vessel, 0.0);

                  // Apply flux to degrees of freedom
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const unsigned int component_i =
                        fe.system_to_component_index(i).first;

                      // DG flux integral: ∫_face flux * v * dS
                      // Sign depends on face orientation (outward normal)
                      const double flux_contribution =
                        numerical_flux[component_i] * normal[0];

                      cell_rhs[i] += flux_contribution *
                                     fe_face_values.shape_value(i, q) *
                                     fe_face_values.JxW(q);
                    }
                }
            }

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              if (dof_handler.locally_owned_dofs().is_element(
                    local_dof_indices[i]))
                {
                  system_rhs[local_dof_indices[i]] -= cell_rhs[i];
                }
            }
        }
    }

  system_rhs.compress(VectorOperation::add);
}

template <int spacedim>
// Python equivalent: numerics.py::computeNewSolution solve phase (lines
// 260-280)
void
BloodFlow1D<spacedim>::solve_time_step()
{
  // For explicit time stepping, we directly update the solution
  // In a more sophisticated implementation, this could use implicit methods

  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();

  for (auto it = locally_owned_dofs.begin(); it != locally_owned_dofs.end();
       ++it)
    {
      solution[*it] = old_solution[*it] - time_step * system_rhs[*it];
    }

  solution.compress(VectorOperation::insert);

  // Apply slope limiting
  apply_slope_limiting();

  // Update old solution
  old_solution = solution;

  // Compute primitive variables for output
  compute_primitive_variables();
}

template <int spacedim>
// Python equivalent: vessel.py::dQ slope limiting (line 27)
void
BloodFlow1D<spacedim>::apply_slope_limiting()
{
  // Implement slope limiting for stability
  // For now, just ensure positive area
  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const unsigned int vessel_id =
            cell_to_vessel_map[cell->global_active_cell_index()];
          const VesselData &vessel = vessel_data[vessel_id];

          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              const unsigned int component =
                fe.system_to_component_index(i).first;
              const types::global_dof_index global_dof = cell->dof_index(i);

              if (locally_owned_dofs.is_element(global_dof))
                {
                  if (component == 0) // Area component
                    {
                      // Ensure area remains positive and within reasonable
                      // bounds
                      solution[global_dof] =
                        std::max(solution[global_dof],
                                 0.1 * vessel.reference_area);
                      solution[global_dof] =
                        std::min(solution[global_dof],
                                 10.0 * vessel.reference_area);
                    }
                }
            }
        }
    }

  solution.compress(VectorOperation::insert);
}

// Python equivalent: modelBloodFlow.py::model.physicalFlux (line 140)
template <int spacedim>
Vector<double>
BloodFlow1D<spacedim>::compute_flux_function(const Vector<double> &state,
                                             const VesselData     &vessel_data,
                                             const double /*x_position*/) const
{
  Vector<double> flux(2);

  const double A = state[0];
  const double Q = state[1];

  // Continuity equation flux: F₁ = Q
  flux[0] = Q;

  // Momentum equation flux: F₂ = Q²/A + ∫p dA
  // Following Python implementation: F[1] =
  // a*u*u+k*a/rho*(m/(m+1.)*(a/a0)**m-n/(n+1.)*(a/a0)**n)
  if (A > 1e-12)
    {
      const double u     = Q / A; // Average velocity
      const double A0    = vessel_data.reference_area;
      const double K     = vessel_data.elastic_modulus;
      const double m     = parameters.tube_law_m;
      const double n     = parameters.tube_law_n;
      const double ratio = A / A0;

      // Momentum flux from Python model
      flux[1] = A * u * u + K * A / parameters.rho *
                              (m / (m + 1.0) * std::pow(ratio, m) -
                               n / (n + 1.0) * std::pow(ratio, n));
    }
  else
    {
      flux[1] = 0.0;
    }

  return flux;
}

// Python equivalent: modelBloodFlow.py::model.pFa (line 95)
template <int spacedim>
double
BloodFlow1D<spacedim>::compute_pressure(const double      area,
                                        const VesselData &vessel_data) const
{
  const double A0    = vessel_data.reference_area;
  const double ratio = area / A0;

  // Tube law from Python: p = K * ( (a/a0)^m - (a/a0)^n ) + p0 + pe
  // For simplified case with n=0, m=0.5: p = K * (sqrt(A/A0) - 1) + p0
  const double K = vessel_data.elastic_modulus;
  const double m = parameters.tube_law_m;
  const double n = parameters.tube_law_n;

  const double pressure = K * (std::pow(ratio, m) - std::pow(ratio, n)) +
                          parameters.reference_pressure;

  return pressure;
}

// Python equivalent: modelBloodFlow.py::model.waveSpeed (line 117)
template <int spacedim>
double
BloodFlow1D<spacedim>::compute_wave_speed_at_state(
  const Vector<double> &state,
  const VesselData     &vessel_data) const
{
  const double A     = state[0];
  const double A0    = vessel_data.reference_area;
  const double ratio = A / A0;

  // Wave speed from Python: c = sqrt(a/rho * dpda)
  // where dpda = K*(m*a^(m-1)/a0^m - n*a^(n-1)/a0^n)
  const double K = vessel_data.elastic_modulus;
  const double m = parameters.tube_law_m;
  const double n = parameters.tube_law_n;

  // dpda = K*(m*(A/A0)^(m-1)/A0 - n*(A/A0)^(n-1)/A0)
  const double dpda =
    K * (m * std::pow(ratio, m - 1.0) / A0 - n * std::pow(ratio, n - 1.0) / A0);

  const double c = std::sqrt(A / parameters.rho * dpda);

  return c;
}

// Python equivalent: modelBloodFlow.py::model.riemannInvariants (line 578)
template <int spacedim>
std::pair<double, double>
BloodFlow1D<spacedim>::compute_riemann_invariants(
  const Vector<double> &state,
  const VesselData     &vessel_data) const
{
  const double A = state[0];
  const double Q = state[1];
  const double u = Q / A;

  // Following Python: gri1= u+gamma*2./m*a**(m/2.)
  //                   gri2= u-gamma*2./m*a**(m/2.)
  const double A0    = vessel_data.reference_area;
  const double K     = vessel_data.elastic_modulus;
  const double m     = parameters.tube_law_m;
  const double gamma = std::sqrt(K * m / parameters.rho / std::pow(A0, m));

  const double riemann_term = gamma * 2.0 / m * std::pow(A, m / 2.0);

  const double gri1 = u + riemann_term;
  const double gri2 = u - riemann_term;

  return std::make_pair(gri1, gri2);
}

// Python equivalent: modelBloodFlow.py::model.riemannInvariantIntegral (line
// 267)
template <int spacedim>
double
BloodFlow1D<spacedim>::riemann_invariant_integral(
  const double      area_left,
  const double      area_right,
  const VesselData &vessel_data) const
{
  // Transform to proper deal.II integration using quadrature
  // Integral: ∫[area_left to area_right] 2*c(A)/m * dA
  // where c(A) = sqrt(A/rho * K*m*(A/A0)^(m-1)/A0)

  const double m = parameters.tube_law_m;
  // const double n = 0.0; // Parameter from constitutive law (not used in
  // integration)

  // Use deal.II quadrature for numerical integration
  const QGauss<1>    quadrature(fe.degree + 2); // Higher order for accuracy
  const unsigned int n_q_points = quadrature.size();

  double integral_result = 0.0;

  // Map integration interval [area_left, area_right] to [0,1]
  const double interval_length = area_right - area_left;

  // Handle the case where areas are equal
  if (std::abs(interval_length) < 1e-12)
    return 0.0;

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Map quadrature point from [0,1] to [area_left, area_right]
      const double xi     = quadrature.point(q)[0]; // Quadrature point in [0,1]
      const double area   = area_left + xi * interval_length;
      const double weight = quadrature.weight(q);

      // Ensure area is positive
      if (area <= 1e-12)
        continue;

      // Create state vector for wave speed computation
      Vector<double> state(2);
      state[0] = area;
      state[1] = 0.0; // Flow rate doesn't affect wave speed calculation

      // Compute wave speed at this area
      const double c = compute_wave_speed_at_state(state, vessel_data);

      // Integrand: 2*c(A)/m
      const double integrand = 2.0 * c / m;

      // Add contribution to integral (including Jacobian of transformation)
      integral_result += integrand * weight * interval_length;
    }

  return integral_result;
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py primitive variables setup (line 145)
void
BloodFlow1D<spacedim>::compute_primitive_variables()
{
  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const unsigned int vessel_id =
            cell_to_vessel_map[cell->global_active_cell_index()];
          const VesselData &vessel = vessel_data[vessel_id];

          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              const unsigned int component =
                fe.system_to_component_index(i).first;
              const types::global_dof_index global_dof = cell->dof_index(i);

              if (locally_owned_dofs.is_element(global_dof))
                {
                  if (component == 0) // Area component
                    {
                      const double A       = solution[global_dof];
                      pressure[global_dof] = compute_pressure(A, vessel);
                    }
                  else // Flow rate component
                    {
                      const double Q = solution[global_dof];
                      // Find corresponding area DOF
                      const unsigned int area_dof =
                        cell->dof_index(i - 1); // Assumes area comes first
                      const double A = solution[area_dof];

                      if (A > 1e-12)
                        {
                          velocity[global_dof] = Q / A;
                        }
                      else
                        {
                          velocity[global_dof] = 0.0;
                        }
                    }
                }
            }
        }
    }

  pressure.compress(VectorOperation::insert);
  velocity.compress(VectorOperation::insert);
}

template <int spacedim>
// Python equivalent: Physical constraints checking (implicit in Python
// numerics)
void
BloodFlow1D<spacedim>::check_physical_constraints()
{
  // Check for negative areas or other unphysical values
  double min_area = std::numeric_limits<double>::max();
  double max_area = 0.0;

  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              const unsigned int component =
                fe.system_to_component_index(i).first;
              const types::global_dof_index global_dof = cell->dof_index(i);

              if (locally_owned_dofs.is_element(global_dof) && component == 0)
                {
                  const double area = solution[global_dof];
                  min_area          = std::min(min_area, area);
                  max_area          = std::max(max_area, area);
                }
            }
        }
    }

  min_area = Utilities::MPI::min(min_area, mpi_communicator);
  max_area = Utilities::MPI::max(max_area, mpi_communicator);

  if (min_area < 0.0)
    {
      pcout << "Warning: Negative area detected: " << min_area << std::endl;
    }

  if (timestep_number % (10 * parameters.output_frequency) == 0)
    {
      pcout << "  Area range: [" << min_area << ", " << max_area << "]"
            << std::endl;
    }
}

template <int spacedim>
// Python equivalent: Output function (implicit in main_bfe_network.py plotting,
// line 875)
void
BloodFlow1D<spacedim>::output_results() const
{
  TimerOutput::Scope t(computing_timer, "output");

  static std::vector<std::pair<double, std::string>> time_and_solutions;

  DataOut<1, spacedim> data_out;
  data_out.attach_dof_handler(dof_handler);

  // Add solution components
  std::vector<std::string> solution_names(2);
  solution_names[0] = "area";
  solution_names[1] = "flow_rate";

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(2,
                             DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<1, spacedim>::type_dof_data,
                           component_interpretation);

  // Add primitive variables
  data_out.add_data_vector(pressure, "pressure");
  data_out.add_data_vector(velocity, "velocity");

  // Add vessel properties as cell data
  Vector<double> vessel_ids(triangulation.n_active_cells());
  Vector<double> reference_areas(triangulation.n_active_cells());
  Vector<double> wave_speeds(triangulation.n_active_cells());

  unsigned int cell_index = 0;
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const unsigned int vessel_id =
            cell_to_vessel_map.at(cell->global_active_cell_index());
          const VesselData &vessel = vessel_data[vessel_id];

          vessel_ids[cell_index]      = vessel_id;
          reference_areas[cell_index] = vessel.reference_area;
          wave_speeds[cell_index]     = vessel.wave_speed;
          ++cell_index;
        }
    }

  data_out.add_data_vector(vessel_ids, "vessel_id");
  data_out.add_data_vector(reference_areas, "reference_area");
  data_out.add_data_vector(wave_speeds, "wave_speed");

  data_out.build_patches();

  const std::string filename =
    parameters.output_basename + "_" + std::to_string(output_number) + ".vtu";
  const std::string full_filename = parameters.output_directory + filename;

  std::ofstream output(full_filename);
  data_out.write_vtu(output);

  // Write master file for parallel visualization
  if (this_mpi_process == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0; i < n_mpi_processes; ++i)
        {
          filenames.push_back(filename);
        }

      const std::string master_filename =
        parameters.output_directory + parameters.output_basename + "_" +
        std::to_string(output_number) + ".pvtu";
      std::ofstream master_output(master_filename);
      data_out.write_pvtu_record(master_output, filenames);

      // Add to time series for PVD output
      time_and_solutions.push_back({current_time, filename});

      // Write PVD file
      const std::string pvd_filename =
        parameters.output_directory + parameters.output_basename + ".pvd";
      std::ofstream pvd_output(pvd_filename);
      DataOutBase::write_pvd_record(pvd_output, time_and_solutions);
    }

  ++output_number;
}

// Python equivalent: modelBloodFlow.py::model.prescribeInletFlow (line 769)
template <int spacedim>
std::pair<double, double>
BloodFlow1D<spacedim>::prescribe_inlet_flow(const double      area_1d,
                                            const double      flow_1d,
                                            const double      flow_bc,
                                            const VesselData &vessel_data) const
{
  // Following Python prescribeInletFlow function
  const double   m = 0.5; // Assuming simplified case
  Vector<double> state_1d(2);
  state_1d[0]      = area_1d;
  state_1d[1]      = flow_1d;
  const double c1D = compute_wave_speed_at_state(state_1d, vessel_data);

  // Function to solve for prescribing inlet flow
  // qBC/aBC - q1D/a1D + 2./m*c1D - 2./m*cBC = 0
  // Solve iteratively for aBC
  double       aBC      = area_1d; // Initial guess
  const double tol      = 1e-12;
  const int    max_iter = 100;

  for (int i = 0; i < max_iter; ++i)
    {
      Vector<double> state_bc(2);
      state_bc[0]      = aBC;
      state_bc[1]      = 0.0;
      const double cBC = compute_wave_speed_at_state(state_bc, vessel_data);
      const double f =
        flow_bc / aBC - flow_1d / area_1d + 2.0 / m * c1D - 2.0 / m * cBC;

      if (std::abs(f) < tol)
        break;

      // Simple Newton iteration (simplified)
      const double   eps = 1e-10;
      Vector<double> state_bc_plus(2);
      state_bc_plus[0] = aBC + eps;
      state_bc_plus[1] = 0.0;
      const double cBC_plus =
        compute_wave_speed_at_state(state_bc_plus, vessel_data);
      const double f_plus = flow_bc / (aBC + eps) - flow_1d / area_1d +
                            2.0 / m * c1D - 2.0 / m * cBC_plus;
      const double df = (f_plus - f) / eps;

      aBC = aBC - f / df;
    }

  return std::make_pair(aBC, flow_bc);
}

// Python equivalent: modelBloodFlow.py::model.prescribeInletPressure (line 793)
template <int spacedim>
std::pair<double, double>
BloodFlow1D<spacedim>::prescribe_inlet_pressure(
  const double      area_1d,
  const double      flow_1d,
  const double      pressure_bc,
  const VesselData &vessel_data) const
{
  // Following Python prescribeInletPressure function
  const double m = 0.5; // Assuming simplified case

  // Compute area from pressure using inverse of tube law
  const double A0    = vessel_data.reference_area;
  const double K     = vessel_data.elastic_modulus;
  const double ratio = (pressure_bc - parameters.reference_pressure) / K + 1.0;
  const double aBC   = A0 * std::pow(ratio, 1.0 / m);

  // Compute corresponding flow
  Vector<double> state_1d(2);
  state_1d[0]      = area_1d;
  state_1d[1]      = flow_1d;
  const double c1D = compute_wave_speed_at_state(state_1d, vessel_data);

  Vector<double> state_bc(2);
  state_bc[0]      = aBC;
  state_bc[1]      = 0.0;
  const double cBC = compute_wave_speed_at_state(state_bc, vessel_data);
  const double qBC = (flow_1d / area_1d - 2.0 / m * c1D + 2.0 / m * cBC) * aBC;

  return std::make_pair(aBC, qBC);
}

// Python equivalent: modelBloodFlow.py::model.inflowAorta (line 743)
template <int spacedim>
double
BloodFlow1D<spacedim>::compute_inflow_function(const double time,
                                               const int    flow_type) const
{
  // Following Python inflowAorta function
  if (flow_type == 1)
    {
      // Boileau et al benchmark paper, aortic bifurcation flow
      const double T = 1.1;
      const double q =
        10e5 *
        (7.9853e-06 + 2.6617e-05 * std::sin(2 * M_PI * time / T + 0.29498) +
         2.3616e-05 * std::sin(4 * M_PI * time / T - 1.1403) -
         1.9016e-05 * std::sin(6 * M_PI * time / T + 0.40435) -
         8.5899e-06 * std::sin(8 * M_PI * time / T - 1.1892) -
         2.436e-06 * std::sin(10 * M_PI * time / T - 1.4918) +
         1.4905e-06 * std::sin(12 * M_PI * time / T + 1.0536) +
         1.3581e-06 * std::sin(14 * M_PI * time / T - 0.47666) -
         6.3031e-07 * std::sin(16 * M_PI * time / T + 0.93768));
      return q;
    }
  else if (flow_type == 0)
    {
      // Boileau et al benchmark paper, thoracic aorta flow
      const double T = 0.9550;
      const double q =
        500 * (0.20617 + 0.37759 * std::sin(2 * M_PI * time / T + 0.59605) +
               0.2804 * std::sin(4 * M_PI * time / T - 0.35859) +
               0.15337 * std::sin(6 * M_PI * time / T - 1.2509) -
               0.049889 * std::sin(8 * M_PI * time / T + 1.3921) +
               0.038107 * std::sin(10 * M_PI * time / T - 1.1068) -
               0.041699 * std::sin(12 * M_PI * time / T + 1.3985));
      return q;
    }

  // Default case - steady flow
  return 0.0;
}

// Python equivalent: modelBloodFlow.py::model.getkfromc0 (line 736)
template <int spacedim>
double
BloodFlow1D<spacedim>::compute_elastic_modulus_from_wave_speed(
  const double wave_speed,
  const double /*reference_area*/) const
{
  // Following Python getkfromc0 function: k = c^2*rho/(m-n)
  const double m  = 0.5; // Assuming simplified case
  const double n  = 0.0; // Assuming simplified case
  const double c2 = wave_speed * wave_speed;
  const double k  = c2 * parameters.rho / (m - n);
  return k;
}

// Python equivalent: modelBloodFlow.py::model.aFp (line 719) - inverse of pFa
// Add function to compute area from pressure (inverse tube law)
template <int spacedim>
double
BloodFlow1D<spacedim>::compute_area_from_pressure(
  const double      pressure,
  const VesselData &vessel_data) const
{
  // Following Python aFp function: a = ((p-p0-pe)/k+1.)**(1./m)*a0
  const double A0 = vessel_data.reference_area;
  const double K  = vessel_data.elastic_modulus;
  const double m  = 0.5; // Assuming simplified case
  const double n  = 0.0; // Assuming simplified case

  if (std::abs(n) > 1e-8)
    {
      // Not implemented for n!=0 case
      return A0;
    }

  const double ratio = (pressure - parameters.reference_pressure) / K + 1.0;
  const double area  = A0 * std::pow(ratio, 1.0 / m);

  return area;
}

template <int spacedim>
// Python equivalent: numerics.py::numFluxLF and numFluxRoe (lines 85-110)
Vector<double>
BloodFlow1D<spacedim>::compute_hll_flux_vector(
  const Vector<double> &state_left,
  const Vector<double> &state_right,
  const VesselData     &vessel_data,
  const double /*x_position*/) const
{
  // Implement HLL numerical flux vector using proper deal.II integration
  const double A_L = state_left[0];
  const double Q_L = state_left[1];
  const double A_R = state_right[0];
  const double Q_R = state_right[1];

  Vector<double> F_HLL(2);

  // Avoid division by zero
  if (A_L <= 1e-12 || A_R <= 1e-12)
    {
      F_HLL[0] = 0.0;
      F_HLL[1] = 0.0;
      return F_HLL;
    }

  const double u_L = Q_L / A_L;
  const double u_R = Q_R / A_R;

  // Compute wave speeds using proper deal.II approach
  const double c_L = compute_wave_speed_at_state(state_left, vessel_data);
  const double c_R = compute_wave_speed_at_state(state_right, vessel_data);

  // Toro's wave speed estimates for HLL solver
  const double S_L = std::min(u_L - c_L, u_R - c_R);
  const double S_R = std::max(u_L + c_L, u_R + c_R);

  // Compute physical fluxes
  const Vector<double> F_L =
    compute_flux_function(state_left, vessel_data, 0.0);
  const Vector<double> F_R =
    compute_flux_function(state_right, vessel_data, 0.0);

  for (unsigned int comp = 0; comp < 2; ++comp)
    {
      if (S_L >= 0.0)
        {
          F_HLL[comp] = F_L[comp];
        }
      else if (S_R <= 0.0)
        {
          F_HLL[comp] = F_R[comp];
        }
      else
        {
          // HLL flux formula
          F_HLL[comp] = (S_R * F_L[comp] - S_L * F_R[comp] +
                         S_L * S_R * (state_right[comp] - state_left[comp])) /
                        (S_R - S_L);
        }
    }

  return F_HLL;
}

template <int spacedim>
// Python equivalent: numerics.py::numFluxLF scalar component (lines 85-110)
double
BloodFlow1D<spacedim>::compute_hll_flux(const Vector<double> &state_left,
                                        const Vector<double> &state_right,
                                        const VesselData     &vessel_data,
                                        const double /*x_position*/) const
{
  // Return scalar flux for specific component (kept for compatibility)
  const Vector<double> flux_vector =
    compute_hll_flux_vector(state_left, state_right, vessel_data, 0.0);
  return flux_vector[0]; // Return first component
}

// Helper functions for boundary conditions
template <int spacedim>
// Python equivalent: modelBloodFlow.py boundary condition functions (lines
// 769-850)
Vector<double>
BloodFlow1D<spacedim>::apply_inlet_bc(const Vector<double> &state_interior,
                                      const VesselData     &vessel_data,
                                      const double          time) const
{
  Vector<double> state_bc(2);

  // Apply inlet boundary condition (flow or pressure)
  if (vessel_data.inlet_bc_type == 0) // Prescribed flow
    {
      const double flow_bc   = compute_inlet_flow_rate(time);
      const auto   bc_result = prescribe_inlet_flow(state_interior[0],
                                                  state_interior[1],
                                                  flow_bc,
                                                  vessel_data);
      state_bc[0]            = bc_result.first;  // area
      state_bc[1]            = bc_result.second; // flow
    }
  else // Prescribed pressure
    {
      const double pressure_bc = parameters.reference_pressure; // Simplified
      const auto   bc_result   = prescribe_inlet_pressure(state_interior[0],
                                                      state_interior[1],
                                                      pressure_bc,
                                                      vessel_data);
      state_bc[0]              = bc_result.first;  // area
      state_bc[1]              = bc_result.second; // flow
    }

  return state_bc;
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py outlet boundary conditions (implicit)
Vector<double>
BloodFlow1D<spacedim>::apply_outlet_bc(const Vector<double> &state_interior,
                                       const VesselData     &vessel_data,
                                       const double /*time*/) const
{
  Vector<double> state_bc(2);

  // Apply outlet boundary condition (typically pressure or resistance)
  if (vessel_data.outlet_bc_type == 0) // Prescribed pressure
    {
      const double pressure_bc = parameters.reference_pressure; // Simplified
      const auto   bc_result   = prescribe_inlet_pressure(state_interior[0],
                                                      state_interior[1],
                                                      pressure_bc,
                                                      vessel_data);
      state_bc[0]              = bc_result.first;  // area
      state_bc[1]              = bc_result.second; // flow
    }
  else // No boundary condition (copy interior state)
    {
      state_bc = state_interior;
    }

  return state_bc;
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py::inflowAorta inlet flow computation
// (line 743)
double
BloodFlow1D<spacedim>::compute_inlet_flow_rate(const double time) const
{
  // Use the inflow function to compute time-dependent inlet flow rate
  // Default to flow_type = 1 (aortic bifurcation) for now
  // This could be made configurable via parameters in the future
  return compute_inflow_function(time, 1);
}

template <int spacedim>
// Python equivalent: Mesh connectivity (implicit in main_bfe_network.py vessel
// network setup)
void
BloodFlow1D<spacedim>::create_face_connectivity_map()
{
  // Clear any existing connectivity map
  face_to_cells_map.clear();

  pcout << "  Creating face-to-cells connectivity map..." << std::endl;

  // Iterate over all locally owned cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          // Iterate over all faces of this cell
          for (const auto face_number : cell->face_indices())
            {
              const auto face = cell->face(face_number);

              // Add this cell to the face's connectivity list
              face_to_cells_map[face].push_back(cell);
            }
        }
    }

  pcout << "  Face connectivity map created with " << face_to_cells_map.size()
        << " unique faces." << std::endl;
}

// Missing function implementations

template <int spacedim>
// Python equivalent: modelBloodFlow.py::model.dpda (line 106)
double
BloodFlow1D<spacedim>::compute_pressure_derivative(
  const double      area,
  const VesselData &vessel_data) const
{
  // Python: k*(m*a**(m-1.)/a0**m-n*a**(n-1.)/a0**n)
  const double k  = vessel_data.elastic_modulus;
  const double a0 = vessel_data.reference_area;
  const double m  = parameters.tube_law_m;
  const double n  = parameters.tube_law_n;

  const double a_over_a0 = area / a0;
  return k * (m * std::pow(a_over_a0, m - 1.0) / a0 -
              n * std::pow(a_over_a0, n - 1.0) / a0);
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py::model.lambdaMatrix (line 219)
FullMatrix<double>
BloodFlow1D<spacedim>::compute_eigenvalue_matrix(
  const Vector<double> &state,
  const VesselData     &vessel_data) const
{
  // Python: lambdaMat[0,0] = u-c; lambdaMat[1,1] = u+c
  const double area     = state[0];
  const double velocity = state[1] / state[0];
  const double c        = compute_wave_speed_at_state(state, vessel_data);

  FullMatrix<double> lambda_matrix(2, 2);
  lambda_matrix[0][0] = velocity - c;
  lambda_matrix[1][1] = velocity + c;
  // Off-diagonal elements remain zero
  (void)area; // Suppress unused warning - area might be used in more complex
              // tube laws

  return lambda_matrix;
}

template <int spacedim>
// Python equivalent: vessel.py::vessel.setInitialCondition (line 98)
void
BloodFlow1D<spacedim>::set_initial_conditions(const double initial_pressure,
                                              const double initial_velocity)
{
  // Python: for i in range(self.nCells):
  //           self.Q[i,0] = self.mod.aFp(pIni,self.xC[i])
  //           self.Q[i,1] = uIni*self.Q[i,0]

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      const unsigned int vessel_id =
        cell_to_vessel_map[cell->global_active_cell_index()];
      const VesselData &vessel = vessel_data[vessel_id];

      // Compute area from pressure using inverse tube law
      const double area = compute_area_from_pressure(initial_pressure, vessel);
      const double flow = initial_velocity * area;

      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        {
          const unsigned int component = fe.system_to_component_index(i).first;
          if (component == 0) // Area component
            solution[dof_indices[i]] = area;
          else if (component == 1) // Flow component
            solution[dof_indices[i]] = flow;
        }
    }

  solution.compress(VectorOperation::insert);
}

template <int spacedim>
// Python equivalent: numerics.py::numericalMethod.update (line 208)
void
BloodFlow1D<spacedim>::update_solution_time_step()
{
  // Python: Main time stepping update with flux integration
  // This replaces/extends the existing assemble_system functionality

  // Store old solution
  old_solution = solution;

  // Reset system
  system_rhs = 0;

  // Update boundary conditions
  // (simplified - full implementation would iterate over vessels)

  // Assemble the time update
  assemble_system();

  // Apply time step update (explicit Euler for now)
  for (unsigned int i = 0; i < solution.size(); ++i)
    {
      if (solution.locally_owned_elements().is_element(i))
        solution[i] = old_solution[i] + time_step * system_rhs[i];
    }

  solution.compress(VectorOperation::insert);
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py::model.prescribeRCR (referenced in
// vessel.py:203)
Vector<double>
BloodFlow1D<spacedim>::prescribe_rcr_boundary_condition(
  const Vector<double> &interior_state,
  const double          rcr_pressure,
  const double          rcr_resistance,
  const VesselData     &vessel_data) const
{
  // Python: aBC, qBC =
  // self.mod.prescribeRCR(Q1D[0],Q1D[1],self.pRCR,self.r1,self.length) This is
  // a simplified implementation - full RCR requires more complex boundary
  // treatment

  Vector<double> boundary_state(2);

  // Extract interior state
  const double A_interior = interior_state[0];
  const double Q_interior = interior_state[1];

  // Apply RCR boundary condition
  // Simplified: use pressure matching with resistance
  const double pressure_interior   = compute_pressure(A_interior, vessel_data);
  const double pressure_difference = pressure_interior - rcr_pressure;
  const double Q_boundary          = pressure_difference / rcr_resistance;

  // Area boundary condition from characteristic analysis
  const double c = compute_wave_speed_at_state(interior_state, vessel_data);
  const double u_interior = Q_interior / A_interior;

  // Characteristic: W2 = u + 2*c is constant across boundary
  const double W2         = u_interior + 2.0 * c;
  const double u_boundary = Q_boundary / A_interior; // Approximate
  const double A_boundary = A_interior;              // Simplified

  // TODO: Implement full characteristic analysis using W2 and u_boundary
  (void)W2;
  (void)u_boundary; // Suppress unused warnings for placeholder implementation

  boundary_state[0] = A_boundary;
  boundary_state[1] = Q_boundary;

  return boundary_state;
}

template <int spacedim>
// Python equivalent: vessel.py::vessel.setBoundaryConditions (line 188)
void
BloodFlow1D<spacedim>::set_boundary_conditions_for_vessel(
  const unsigned int vessel_id,
  const double       time)
{
  // Python: Complex boundary condition handling with multiple types
  // This is a simplified implementation focusing on essential BC types

  (void)vessel_id; // Suppress unused parameter warning
  (void)time;      // Suppress unused parameter warning

  // TODO: Implement full boundary condition handling
  // For now, this is a placeholder that can be extended
}

template <int spacedim>
// Python equivalent: numerics.py::numericalMethod.reconstructionENO (line 302)
void
BloodFlow1D<spacedim>::apply_eno_reconstruction()
{
  // Python: High-order ENO reconstruction for improved accuracy
  // This is a placeholder for the complex ENO algorithm

  // TODO: Implement ENO reconstruction
  // For now, this is a no-op - can be extended with proper ENO stencils
}

template <int spacedim>
// Python equivalent: vessel.py::vessel.getMusclHancockEvolveState (line 170)
void
BloodFlow1D<spacedim>::apply_muscl_reconstruction()
{
  // Python: MUSCL-Hancock reconstruction for higher-order accuracy
  // This is a placeholder for MUSCL reconstruction

  // TODO: Implement MUSCL reconstruction with slope limiters
  // For now, this is a no-op - can be extended with proper MUSCL stencils
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py::model.source (referenced in vessel.py)
Vector<double>
BloodFlow1D<spacedim>::compute_source_term(const Vector<double> &state,
                                           const VesselData     &vessel_data,
                                           const double /*time*/) const
{
  // Python: Source terms including friction and geometric effects
  Vector<double> source(2);

  const double area     = state[0];
  const double flow     = state[1];
  const double velocity = flow / area;

  // Friction source term: -f*Q*|Q|/(2*A*rho) where f is friction factor
  const double friction_factor = parameters.vel_profile_coeff * parameters.mu /
                                 (parameters.rho * vessel_data.reference_area);

  source[0] = 0.0; // No source for area equation
  source[1] = -friction_factor * flow * std::abs(flow) /
              (2.0 * area); // Friction for momentum

  // TODO: Add geometric source terms based on vessel curvature/tapering if
  // needed
  (void)velocity; // Suppress unused warning - velocity may be used in future
                  // extensions

  return source;
}

template <int spacedim>
// Python equivalent: numerics.py::numericalMethod.numFluxLW (line 97)
Vector<double>
BloodFlow1D<spacedim>::compute_lax_wendroff_flux(
  const Vector<double> &state_left,
  const Vector<double> &state_right,
  const VesselData     &vessel_data,
  const double          dx,
  const double          dt) const
{
  // Python: FL = self.mod.physicalFlux(QL,x)
  //         FR = self.mod.physicalFlux(QR,x)
  //         QLW = 0.5 * (QL + QR) - 0.5 * dt/dx * (FR-FL)
  //         fLW = self.mod.physicalFlux(QLW,x)

  const Vector<double> F_L =
    compute_flux_function(state_left, vessel_data, 0.0);
  const Vector<double> F_R =
    compute_flux_function(state_right, vessel_data, 0.0);

  Vector<double> state_lw(2);
  for (unsigned int comp = 0; comp < 2; ++comp)
    {
      state_lw[comp] = 0.5 * (state_left[comp] + state_right[comp]) -
                       0.5 * dt / dx * (F_R[comp] - F_L[comp]);
    }

  return compute_flux_function(state_lw, vessel_data, 0.0);
}

template <int spacedim>
// Python equivalent: numerics.py::numericalMethod.numFluxFORCE (line 108)
Vector<double>
BloodFlow1D<spacedim>::compute_force_flux(const Vector<double> &state_left,
                                          const Vector<double> &state_right,
                                          const VesselData     &vessel_data,
                                          const double          dx,
                                          const double          dt) const
{
  // Python: fLF = self.numFluxLF(dx,dt,QL,QR,x)
  //         fLW = self.numFluxLW(dx,dt,QL,QR,x)
  //         fFORCE = 0.5 * (fLF + fLW)

  // Lax-Friedrichs flux
  const Vector<double> F_L =
    compute_flux_function(state_left, vessel_data, 0.0);
  const Vector<double> F_R =
    compute_flux_function(state_right, vessel_data, 0.0);

  Vector<double> flux_lf(2);
  for (unsigned int comp = 0; comp < 2; ++comp)
    {
      flux_lf[comp] = 0.5 * (F_L[comp] + F_R[comp]) -
                      0.5 * dx / dt * (state_right[comp] - state_left[comp]);
    }

  // Lax-Wendroff flux
  const Vector<double> flux_lw =
    compute_lax_wendroff_flux(state_left, state_right, vessel_data, dx, dt);

  // FORCE flux (average)
  Vector<double> flux_force(2);
  for (unsigned int comp = 0; comp < 2; ++comp)
    {
      flux_force[comp] = 0.5 * (flux_lf[comp] + flux_lw[comp]);
    }

  return flux_force;
}

template <int spacedim>
// Python equivalent: vessel.py::vessel.evolveRCR (line 155)
void
BloodFlow1D<spacedim>::evolve_rcr_boundary_condition(
  const double flow_rate,
  const double time_step,
  double      &rcr_pressure,
  const double rcr_capacitance,
  const double rcr_distal_resistance,
  const double rcr_distal_pressure) const
{
  // Python: self.qinRCR = qin
  //         self.qoutRCR = (self.pRCR-self.pRCRdistal)/self.r2
  //         self.pRCR += self.dt/self.c*(self.qinRCR-self.qoutRCR)

  const double q_in = flow_rate;
  const double q_out =
    (rcr_pressure - rcr_distal_pressure) / rcr_distal_resistance;

  rcr_pressure += time_step / rcr_capacitance * (q_in - q_out);
}

template <int spacedim>
// Python equivalent: Used in numerics.py::numericalMethod.numFluxHLL (line 145)
std::pair<double, double>
BloodFlow1D<spacedim>::estimate_wave_speeds_for_hll(
  const Vector<double> &state_left,
  const Vector<double> &state_right,
  const VesselData     &vessel_data) const
{
  // Python: Estimates used in HLL solver for wave speeds
  const double u_L = state_left[1] / state_left[0];
  const double u_R = state_right[1] / state_right[0];

  const double c_L = compute_wave_speed_at_state(state_left, vessel_data);
  const double c_R = compute_wave_speed_at_state(state_right, vessel_data);

  // Toro's estimates
  const double S_L = std::min(u_L - c_L, u_R - c_R);
  const double S_R = std::max(u_L + c_L, u_R + c_R);

  return std::make_pair(S_L, S_R);
}

template <int spacedim>
// Python equivalent: vessel.py::vessel.getSolution (line 115)
Vector<double>
BloodFlow1D<spacedim>::get_cell_average_state(
  const typename DoFHandler<1, spacedim>::active_cell_iterator &cell) const
{
  // Python: linear interpolation of state vector at x
  Vector<double> cell_state(2);

  std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
  cell->get_dof_indices(dof_indices);

  // Get cell average (assuming DG0 or taking first DoF for DG1)
  cell_state[0] = solution[dof_indices[0]]; // Area
  cell_state[1] = solution[dof_indices[1]]; // Flow

  return cell_state;
}

// Python equivalent: modelBloodFlow.py::model.exactSolution (line 658)
template <int spacedim>
Vector<double>
BloodFlow1D<spacedim>::compute_exact_solution(const Point<spacedim> &point,
                                              const double           time) const
{
  // Python: Exact solution for convergence test with sinusoidal wave
  // r0 = 9.99e-3; a0 = np.pi*r0**2; Atilde = a0; L = 1.; T0 = 1.0
  // atilde = 0.1*Atilde; qtilde = 0.
  // exact[:,0] = Atilde + atilde * np.sin(2*np.pi/L*x)*np.cos(2*np.pi/T0*t)
  // exact[:,1] = qtilde - atilde*L/T0 *
  // np.cos(2*np.pi/L*x)*np.sin(2*np.pi/T0*t)

  const double r0      = 9.99e-3;        // Reference radius [m]
  const double a0      = M_PI * r0 * r0; // Reference area [m²]
  const double A_tilde = a0;             // Base area
  const double L       = 1.0;            // Domain length [m]
  const double T0      = 1.0;            // Time period [s]
  const double a_tilde = 0.1 * A_tilde;  // Area amplitude
  const double q_tilde = 0.0;            // Base flow rate

  const double x = point[0]; // Position along vessel

  Vector<double> exact_solution(2);
  exact_solution[0] = A_tilde + a_tilde * std::sin(2.0 * M_PI / L * x) *
                                  std::cos(2.0 * M_PI / T0 * time);
  exact_solution[1] = q_tilde - a_tilde * L / T0 *
                                  std::cos(2.0 * M_PI / L * x) *
                                  std::sin(2.0 * M_PI / T0 * time);

  return exact_solution;
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py::model.source with convergence flag
// (line 691)
Vector<double>
BloodFlow1D<spacedim>::compute_exact_source_term(const Point<spacedim> &point,
                                                 const double time) const
{
  // Python: Source term for convergence test that makes exact solution satisfy
  // PDE
  const double r0      = 9.99e-3;
  const double a0      = M_PI * r0 * r0;
  const double A_tilde = a0;
  const double L       = 1.0;
  const double T0      = 1.0;
  const double a_tilde = 0.1 * A_tilde;

  // Get vessel parameters (assuming first vessel for simplicity)
  const VesselData &vessel = vessel_data[0];
  const double      k      = vessel.elastic_modulus;
  const double      m      = parameters.tube_law_m;

  // Python: gamma = m*k/self.rho/(m+1)/a0**m
  const double gamma = m * k / parameters.rho / (m + 1.0) / std::pow(a0, m);

  const double x = point[0];

  // Exact solution at this point and time
  const Vector<double> exact = compute_exact_solution(point, time);
  const double         A     = exact[0];
  const double         Q     = exact[1];

  Vector<double> source(2);

  // Time derivatives
  const double dAdt = -a_tilde * (2.0 * M_PI / T0) *
                      std::sin(2.0 * M_PI / L * x) *
                      std::sin(2.0 * M_PI / T0 * time);
  const double dQdt = -a_tilde * L / T0 * (2.0 * M_PI / T0) *
                      std::cos(2.0 * M_PI / L * x) *
                      std::cos(2.0 * M_PI / T0 * time);

  // Spatial derivatives
  const double dAdx = a_tilde * (2.0 * M_PI / L) *
                      std::cos(2.0 * M_PI / L * x) *
                      std::cos(2.0 * M_PI / T0 * time);
  const double dQdx = a_tilde * L / T0 * (2.0 * M_PI / L) *
                      std::sin(2.0 * M_PI / L * x) *
                      std::sin(2.0 * M_PI / T0 * time);

  // Pressure derivative with respect to area
  const double dPdA = gamma * std::pow(A / a0, m);

  // Source terms to make exact solution satisfy the PDE
  source[0] = dAdt + dQdx; // Continuity equation residual


  source[1] =
    dQdt + dQdx * Q / A + A * dPdA * dAdx; // Momentum equation residual

  return source;
}

template <int spacedim>
// Python equivalent: modelBloodFlow.py::model.source with convergence flag
// (line 691)
Vector<double>
BloodFlow1D<spacedim>::compute_exact_source_term(const Point<spacedim> &point,
                                                 const double           time,
                                                 const double elastic_modulus,
                                                 const double tube_law_m) const
{
  // Python: Source term for convergence test that makes exact solution satisfy
  // PDE
  const double r0      = 9.99e-3;
  const double a0      = M_PI * r0 * r0;
  const double A_tilde = a0;
  const double L       = 1.0;
  const double T0      = 1.0;
  const double a_tilde = 0.1 * A_tilde;

  // Use provided vessel parameters
  const double k = elastic_modulus;
  const double m = tube_law_m;

  const double gamma = m * k / parameters.rho / (m + 1.0) / std::pow(a0, m);

  const double x = point[0];

  // Exact solution at this point and time
  const Vector<double> exact = compute_exact_solution(point, time);
  const double         A     = exact[0];
  const double         Q     = exact[1];

  Vector<double> source(2);

  // Time derivatives
  const double dAdt = -a_tilde * (2.0 * M_PI / T0) *
                      std::sin(2.0 * M_PI / L * x) *
                      std::sin(2.0 * M_PI / T0 * time);
  const double dQdt = -a_tilde * L / T0 * (2.0 * M_PI / T0) *
                      std::cos(2.0 * M_PI / L * x) *
                      std::cos(2.0 * M_PI / T0 * time);

  // Spatial derivatives
  const double dAdx = a_tilde * (2.0 * M_PI / L) *
                      std::cos(2.0 * M_PI / L * x) *
                      std::cos(2.0 * M_PI / T0 * time);
  const double dQdx = a_tilde * L / T0 * (2.0 * M_PI / L) *
                      std::sin(2.0 * M_PI / L * x) *
                      std::sin(2.0 * M_PI / T0 * time);

  // Pressure derivative with respect to area
  const double dPdA = gamma * std::pow(A / a0, m);

  // Source terms to make exact solution satisfy the PDE
  source[0] = dAdt + dQdx; // Continuity equation residual

  source[1] =
    dQdt + dQdx * Q / A + A * dPdA * dAdx; // Momentum equation residual

  return source;
}

template <int spacedim>
// For testing convergence
double
BloodFlow1D<spacedim>::compute_l2_error_against_exact_solution(
  const double time) const
{
  double l2_error_squared = 0.0;
  double area_sum         = 0.0;

  // Loop over all locally owned cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      // Get quadrature rule for integration
      const QGauss<1>       quadrature_formula(fe.degree + 1);
      FEValues<1, spacedim> fe_values(fe,
                                      quadrature_formula,
                                      update_values | update_quadrature_points |
                                        update_JxW_values);
      fe_values.reinit(cell);

      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      // Extract cell solution
      std::vector<Vector<double>> solution_values(quadrature_formula.size(),
                                                  Vector<double>(2));
      fe_values.get_function_values(solution, solution_values);

      // Compute error at quadrature points
      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        {
          const Point<spacedim> &point    = fe_values.quadrature_point(q);
          const Vector<double>   exact    = compute_exact_solution(point, time);
          const Vector<double>  &computed = solution_values[q];

          const double area_error = computed[0] - exact[0];
          const double flow_error = computed[1] - exact[1];

          l2_error_squared +=
            (area_error * area_error + flow_error * flow_error) *
            fe_values.JxW(q);
          area_sum += fe_values.JxW(q);
        }
    }

  // Sum over all MPI processes
  l2_error_squared = Utilities::MPI::sum(l2_error_squared, mpi_communicator);
  area_sum         = Utilities::MPI::sum(area_sum, mpi_communicator);

  return std::sqrt(l2_error_squared / area_sum);
}

// Explicit instantiations
template class BloodFlow1D<1>;
template class BloodFlow1D<2>;
template class BloodFlow1D<3>;
