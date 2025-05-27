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
        TimerOutput::Scope t(computing_timer, "assembly");
        assemble_system();
      }

      {
        TimerOutput::Scope t(computing_timer, "solve");
        solve_time_step();
      }

      current_time += time_step;
      ++timestep_number;

      if (timestep_number % parameters.output_frequency == 0)
        output_results();

      check_physical_constraints();
    }

  computing_timer.print_summary();
  pcout << "Simulation completed." << std::endl;
}

template <int spacedim>
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
      for (unsigned int i = 0; i < n_cells; ++i)
        {
          vessel_ids[i]  = i;
          lengths[i]     = 1.0;
          inlet_radii[i] = outlet_radii[i] = 0.5;
          wave_speeds[i]                   = 500.0;
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
}

template <int spacedim>
void
BloodFlow1D<spacedim>::setup_boundary_conditions()
{
  // Boundary conditions will be applied during assembly
  pcout << "  Boundary conditions set up." << std::endl;
}

template <int spacedim>
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

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();


  Vector<double>                       cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Get solution values at quadrature points
  std::vector<Vector<double>> solution_values(n_q_points, Vector<double>(2));
  std::vector<Vector<double>> old_solution_values(n_q_points,
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

          // Volume integral (time derivative + flux divergence)
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Vector<double>  &U     = solution_values[q];
              const Vector<double>  &U_old = old_solution_values[q];
              const Point<spacedim> &x_q   = fe_values.quadrature_point(q);

              // Compute flux
              const Vector<double> flux =
                compute_flux_function(U, vessel, x_q[0]);

              // Compute source terms (friction)
              Vector<double> source(2);
              source[0] = 0.0; // Continuity equation has no source

              // Momentum equation friction term
              if (U[0] > 1e-12) // Avoid division by zero
                {
                  const double radius = std::sqrt(U[0] / M_PI);
                  source[1] = -2.0 * M_PI * radius * parameters.mu * U[1] /
                              (parameters.vel_profile_coeff * U[0]);
                }
              else
                {
                  source[1] = 0.0;
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;

                  // Time derivative term
                  cell_rhs[i] +=
                    ((U[component_i] - U_old[component_i]) / time_step) *
                    fe_values.shape_value(i, q) * fe_values.JxW(q);

                  // Flux divergence term (using simple finite difference for
                  // now) This should be replaced with proper flux computation
                  // at faces
                  cell_rhs[i] += flux[component_i] *
                                 fe_values.shape_grad(i, q)[0] *
                                 fe_values.JxW(q);

                  // Source term
                  cell_rhs[i] -= source[component_i] *
                                 fe_values.shape_value(i, q) * fe_values.JxW(q);
                }
            }

          // Face integrals (numerical fluxes) - simplified for now
          // This should include proper HLL/Roe flux computation

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

template <int spacedim>
Vector<double>
BloodFlow1D<spacedim>::compute_flux_function(const Vector<double> &state,
                                             const VesselData     &vessel_data,
                                             const double x_position) const
{
  Vector<double> flux(2);

  const double A = state[0];
  const double Q = state[1];

  // Continuity equation flux: F₁ = Q
  flux[0] = Q;

  // Momentum equation flux: F₂ = Q²/A + ∫p dA
  if (A > 1e-12)
    {
      const double u = Q / A; // Average velocity
      const double p = compute_pressure(A, vessel_data);

      flux[1] = Q * u + p * A; // Simplified pressure integral
    }
  else
    {
      flux[1] = 0.0;
    }

  return flux;
}

template <int spacedim>
double
BloodFlow1D<spacedim>::compute_pressure(const double      area,
                                        const VesselData &vessel_data) const
{
  const double A0    = vessel_data.reference_area;
  const double ratio = area / A0;

  // Simplified tube law: p = K * (sqrt(A/A0) - 1) + p0
  const double K = vessel_data.elastic_modulus;
  const double pressure =
    K * (std::sqrt(ratio) - 1.0) + parameters.reference_pressure;

  return pressure;
}

template <int spacedim>
double
BloodFlow1D<spacedim>::compute_wave_speed_at_state(
  const Vector<double> &state,
  const VesselData     &vessel_data) const
{
  const double A  = state[0];
  const double A0 = vessel_data.reference_area;

  // Wave speed: c = sqrt(K/(2ρ) * sqrt(A0/A))
  const double K = vessel_data.elastic_modulus;
  const double c = std::sqrt(K / (2.0 * parameters.rho) * std::sqrt(A0 / A));

  return c;
}

template <int spacedim>
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
void
BloodFlow1D<spacedim>::output_results() const
{
  TimerOutput::Scope t(const_cast<TimerOutput &>(computing_timer), "output");

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

// Explicit instantiations
template class BloodFlow1D<1>;
template class BloodFlow1D<3>;
