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

#ifndef rdl_blood_flow_1d_h
#define rdl_blood_flow_1d_h

#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "vtk_utils.h"

using namespace dealii;

/**
 * @brief Parameters for the 1D Blood Flow problem
 */
class BloodFlowParameters : public ParameterAcceptor
{
public:
  BloodFlowParameters();

  // Physical parameters
  double rho               = 1.06; // Blood density [g/cm³]
  double mu                = 0.04; // Blood dynamic viscosity [g/(cm·s)]
  double vel_profile_coeff = 22.0; // Velocity profile coefficient (Womersley)

  // Simulation parameters
  double       final_time       = 1.0; // Final simulation time [s]
  double       cfl_number       = 0.9; // CFL number for time stepping
  unsigned int output_frequency = 10;  // Output frequency

  // Mesh parameters
  std::string  mesh_filename = "vessel_network.vtk"; // Input VTK mesh file
  unsigned int fe_degree     = 1;                    // Finite element degree

  // Output parameters
  std::string output_directory = "blood_flow_output/";
  std::string output_basename  = "solution";

  // Numerical scheme parameters
  std::string flux_type =
    "hll"; // Numerical flux type (hll, roe, lax_friedrichs)
  std::string limiter_type = "minmod"; // Slope limiter (minmod, superbee, none)

  // Vessel parameters (will be read from VTK data)
  double reference_pressure = 94666.66; // Reference pressure [g/(cm·s²)]
  
  // Constitutive law parameters 
  double tube_law_m = 0.5; // Constitutive law exponent m
  double tube_law_n = 0.0; // Constitutive law exponent n
  
  // Inflow parameters
  double inlet_flow_amplitude = 500.0; // Inlet flow amplitude scale factor [cm³/s]
  double cardiac_cycle_period = 1.0;   // Cardiac cycle period [s]
  
  // Default vessel parameters (used when data is missing from mesh)
  double default_radius = 0.5;      // Default vessel radius [cm]
  double default_wave_speed = 500.0; // Default wave speed [cm/s]
  double default_wall_thickness = 0.1; // Default wall thickness [cm]
};

/**
 * @brief Vessel structure to hold vessel-specific parameters
 */
struct VesselData
{
  unsigned int inlet_node_id;
  unsigned int outlet_node_id;
  double       length;
  double       inlet_radius;
  double       outlet_radius;
  double       wave_speed;
  unsigned int inlet_bc_type;
  unsigned int outlet_bc_type;
  double       resistance1;
  double       resistance2;
  double       compliance;

  // Derived parameters
  double reference_area;
  double elastic_modulus;
  double wall_thickness;
};

/**
 * @brief 1D Blood Flow Solver using Finite Volume Methods
 *
 * This class implements a 1D blood flow solver based on the hyperbolic system:
 * ∂A/∂t + ∂Q/∂x = 0
 * ∂Q/∂t + ∂(Q²/A + ∫p dA)/∂x = -2π R_0 μ Q/(δ A)
 *
 * where A is the cross-sectional area, Q is the volumetric flow rate,
 * p is the pressure, and the constitutive law is:
 * p = K * ((A/A₀)^m - (A/A₀)^n) + p₀ + p_e
 */
template <int spacedim = 3>
class BloodFlow1D
{
public:
  BloodFlow1D(BloodFlowParameters &parameters);

  void
  run();

private:
  void
  read_mesh_and_data();
  void
  setup_system();
  void
  setup_boundary_conditions();
  void
  create_face_connectivity_map();
  void
  compute_time_step();
  void
  assemble_system();
  void
  solve_time_step();
  void
  output_results() const;

  // Numerical methods
  void
  apply_slope_limiting();
  void
  compute_numerical_flux(const unsigned int face_index);
  Vector<double>
  compute_hll_flux_vector(const Vector<double> &state_left,
                          const Vector<double> &state_right,
                          const VesselData     &vessel_data,
                          const double          x_position) const;
  double
  compute_hll_flux(const Vector<double> &state_left,
                   const Vector<double> &state_right,
                   const VesselData     &vessel_data,
                   const double          x_position) const;

  // Physics functions
  Vector<double>
  compute_flux_function(const Vector<double> &state,
                        const VesselData     &vessel_data,
                        const double          x_position) const;
  double
  compute_pressure(const double area, const VesselData &vessel_data) const;
  double
  compute_wave_speed_at_state(const Vector<double> &state,
                              const VesselData     &vessel_data) const;
  Vector<double>
  compute_eigenvalues(const Vector<double> &state,
                      const VesselData     &vessel_data) const;

  // Boundary conditions
  void
  apply_inlet_boundary_condition(const unsigned int vessel_id,
                                 const double       time);
  void
  apply_outlet_boundary_condition(const unsigned int vessel_id,
                                  const double       time);
  double
  compute_inlet_flow_rate(const double time) const;

  // Helper functions for boundary conditions in face integration
  Vector<double>
  apply_inlet_bc(const Vector<double> &state_interior,
                 const VesselData     &vessel_data,
                 const double          time) const;
  Vector<double>
  apply_outlet_bc(const Vector<double> &state_interior,
                  const VesselData     &vessel_data,
                  const double          time) const;

  // Utility functions
  void
  compute_primitive_variables();
  void
  check_physical_constraints();

  // Additional physics functions from Python model
  std::pair<double, double>
  compute_riemann_invariants(const Vector<double> &state,
                             const VesselData     &vessel_data) const;
  double
  riemann_invariant_integral(const double      area_left,
                             const double      area_right,
                             const VesselData &vessel_data) const;
  std::pair<double, double>
  prescribe_inlet_flow(const double      area_1d,
                       const double      flow_1d,
                       const double      flow_bc,
                       const VesselData &vessel_data) const;
  std::pair<double, double>
  prescribe_inlet_pressure(const double      area_1d,
                           const double      flow_1d,
                           const double      pressure_bc,
                           const VesselData &vessel_data) const;
  double
  compute_inflow_function(const double time, const int flow_type) const;
  double
  compute_elastic_modulus_from_wave_speed(const double wave_speed,
                                          const double reference_area) const;
  double
  compute_area_from_pressure(const double      pressure,
                             const VesselData &vessel_data) const;

  // Matrix computation functions
  FullMatrix<double>
  compute_jacobian(const Vector<double> &state,
                   const VesselData     &vessel_data) const;
  FullMatrix<double>
  compute_right_eigenvector_matrix(const Vector<double> &state,
                                   const VesselData     &vessel_data) const;
  FullMatrix<double>
  compute_right_eigenvector_inverse_matrix(const Vector<double> &state,
                                           const VesselData &vessel_data) const;

  // MPI and parallel data structures
  MPI_Comm            mpi_communicator;
  const unsigned int  n_mpi_processes;
  const unsigned int  this_mpi_process;
  ConditionalOStream  pcout;
  mutable TimerOutput computing_timer;

  // Problem parameters
  BloodFlowParameters &parameters;

  // Mesh and geometry
  parallel::fullydistributed::Triangulation<1, spacedim> triangulation;
  DoFHandler<1, spacedim>                                dof_handler;
  FESystem<1, spacedim> fe; // 2 components: area (A) and flow rate (Q)

  // Vessel network data
  std::vector<VesselData>                         vessel_data;
  std::map<types::global_dof_index, unsigned int> cell_to_vessel_map;

  // Face connectivity map for DG integration
  std::map<typename DoFHandler<1, spacedim>::face_iterator, 
           std::vector<typename DoFHandler<1, spacedim>::active_cell_iterator>> face_to_cells_map;

  // Solution vectors
  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> old_solution;
  LinearAlgebra::distributed::Vector<double> system_rhs;
  LinearAlgebra::distributed::Vector<double> pressure; // For output
  LinearAlgebra::distributed::Vector<double> velocity; // For output

  // Time stepping
  double       current_time;
  double       time_step;
  unsigned int timestep_number;

  // Output
  mutable unsigned int output_number;
};

#endif
