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
 *
 * Python equivalent: modelBloodFlow.py::model.__init__ (line 33)
 * Physical parameters correspond to member variables in model class
 */
class BloodFlowParameters : public ParameterAcceptor
{
public:
  BloodFlowParameters();

  // Physical parameters
  // Python: modelBloodFlow.py::model.rho (line 35)
  double rho = 1.06; // Blood density [g/cm³]
  // Python: modelBloodFlow.py::model.mu (line 38)
  double mu = 0.04; // Blood dynamic viscosity [g/(cm·s)]
  // Python: modelBloodFlow.py::model.vel_profile_coeff (line 40-44)
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
  // Python: modelBloodFlow.py::model.mRef (line 53)
  double tube_law_m = 0.5; // Constitutive law exponent m
  // Python: modelBloodFlow.py::model.nRef (line 54)
  double tube_law_n = 0.0; // Constitutive law exponent n

  // Inflow parameters
  // Python: main_bfe_network.py::scaling factor (line ~100, vessel setup)
  double inlet_flow_amplitude =
    500.0; // Inlet flow amplitude scale factor [cm³/s]
  // Python: main_bfe_network.py::T (line 36)
  double cardiac_cycle_period = 1.0; // Cardiac cycle period [s]

  // Default vessel parameters (used when data is missing from mesh)
  double default_radius         = 0.5;   // Default vessel radius [cm]
  double default_wave_speed     = 500.0; // Default wave speed [cm/s]
  double default_wall_thickness = 0.1;   // Default wall thickness [cm]
};

/**
 * @brief Vessel structure to hold vessel-specific parameters
 *
 * Python equivalent: vessel.py::vessel class (line 4)
 * Individual parameters correspond to mesh data arrays in main_bfe_network.py
 */
struct VesselData
{
  // Python: vessel.py::vessel.iL, iR (line 18-19)
  unsigned int inlet_node_id;
  unsigned int outlet_node_id;
  // Python: vessel.py::vessel.length (line 44)
  double length;
  // Python: main_bfe_network.py::meshData radius columns (line ~110)
  double inlet_radius;
  double outlet_radius;
  // Python: main_bfe_network.py::c (wave speed calculation, line ~115)
  double wave_speed;
  // Python: vessel.py::vessel.bcTypeLeft, bcTypeRight (line 37-38)
  unsigned int inlet_bc_type;
  unsigned int outlet_bc_type;
  // Python: vessel.py::RCRparams or resistance values
  double resistance1;
  double resistance2;
  double compliance;

  // Derived parameters
  // Python: modelBloodFlow.py::model.a0Ref (line 51)
  double reference_area;
  // Python: modelBloodFlow.py::model.kRef (line 50)
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

  // Python equivalent: main_bfe_network.py::main time loop (lines 234-327)
  void
  run();

  /**
   * @brief Compute exact solution for convergence testing
   * Python equivalent: modelBloodFlow.py::model.exactSolution (line 658)
   */
  Vector<double>
  compute_exact_solution(const Point<spacedim> &point, const double time) const;

  /**
   * @brief Compute exact source term for convergence testing
   * Python equivalent: modelBloodFlow.py::model.source with convergence flag
   * (line 691)
   */
  Vector<double>
  compute_exact_source_term(const Point<spacedim> &point,
                            const double           time) const;

  /**
   * @brief Compute exact source term with explicit vessel parameters
   * For testing when vessel_data is not initialized
   */
  Vector<double>
  compute_exact_source_term(const Point<spacedim> &point,
                            const double           time,
                            const double           elastic_modulus,
                            const double           tube_law_m) const;

  /**
   * @brief Check convergence against exact solution
   * For testing purposes
   */
  double
  compute_l2_error_against_exact_solution(const double time) const;

private:
  // Python equivalent: scripts/convert_net_to_vtk.py::read_net_file (line 19)
  void
  read_mesh_and_data();
  // Python equivalent: main_bfe_network.py::vessel setup (lines 129-135)
  void
  setup_system();
  // Python equivalent: boundary condition setup in vessel.py (lines 35-40)
  void
  setup_boundary_conditions();
  // Python equivalent: Mesh connectivity (implicit in main_bfe_network.py
  // vessel network setup)
  void
  create_face_connectivity_map();
  // Python equivalent: numerics.py::computeTimeStep (line 78)
  void
  compute_time_step();
  // Python equivalent: numerics.py::computeNewSolution assembly phase (lines
  // 115-200)
  void
  assemble_system();
  // Python equivalent: numerics.py::computeNewSolution solve phase (lines
  // 260-280)
  void
  solve_time_step();
  // Python equivalent: Output function (implicit in main_bfe_network.py
  // plotting, line 875)
  void
  output_results() const;

  // Numerical methods
  // Python equivalent: vessel.py::dQ slope limiting (line 27)
  void
  apply_slope_limiting();
  void
  compute_numerical_flux(const unsigned int face_index);

  // Python equivalent: numerics.py::numFluxLF and numFluxRoe (lines 85-110)
  Vector<double>
  compute_hll_flux_vector(const Vector<double> &state_left,
                          const Vector<double> &state_right,
                          const VesselData     &vessel_data,
                          const double          x_position) const;
  // Python equivalent: numerics.py::numFluxLF scalar component (lines 85-110)
  double
  compute_hll_flux(const Vector<double> &state_left,
                   const Vector<double> &state_right,
                   const VesselData     &vessel_data,
                   const double          x_position) const;

  // Physics functions
  // Python: modelBloodFlow.py::model.physicalFlux (line 140)
  Vector<double>
  compute_flux_function(const Vector<double> &state,
                        const VesselData     &vessel_data,
                        const double          x_position) const;
  // Python: modelBloodFlow.py::model.pFa (line 95)
  double
  compute_pressure(const double area, const VesselData &vessel_data) const;

  // Python: modelBloodFlow.py::model.waveSpeed (line 117)
  double
  compute_wave_speed_at_state(const Vector<double> &state,
                              const VesselData     &vessel_data) const;

  // Python: modelBloodFlow.py::model.eigenvalues (line 130)
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
  // Python: modelBloodFlow.py::model.inflowAorta (line 743)
  double
  compute_inlet_flow_rate(const double time) const;

  // Helper functions for boundary conditions in face integration
  // Python: modelBloodFlow.py::model.prescribeInletFlow (line 769)
  Vector<double>
  apply_inlet_bc(const Vector<double> &state_interior,
                 const VesselData     &vessel_data,
                 const double          time) const;
  // Python: modelBloodFlow.py::model.prescribeRCR (line 811)
  Vector<double>
  apply_outlet_bc(const Vector<double> &state_interior,
                  const VesselData     &vessel_data,
                  const double          time) const;

  // Utility functions
  // Python equivalent: modelBloodFlow.py primitive variables setup (line 145)
  void
  compute_primitive_variables();

  // Python equivalent: Physical constraints checking (implicit in Python
  // numerics)
  void
  check_physical_constraints();

  // Additional physics functions from Python model
  // Python: modelBloodFlow.py::model.riemannInvariants (line 578)
  std::pair<double, double>
  compute_riemann_invariants(const Vector<double> &state,
                             const VesselData     &vessel_data) const;

  // Python: modelBloodFlow.py::model.riemannInvariantIntegral (line 267)
  double
  riemann_invariant_integral(const double      area_left,
                             const double      area_right,
                             const VesselData &vessel_data) const;

  // Python: modelBloodFlow.py::model.prescribeInletFlow (line 769)
  std::pair<double, double>
  prescribe_inlet_flow(const double      area_1d,
                       const double      flow_1d,
                       const double      flow_bc,
                       const VesselData &vessel_data) const;

  // Python: modelBloodFlow.py::model.prescribeInletPressure (line 793)
  std::pair<double, double>
  prescribe_inlet_pressure(const double      area_1d,
                           const double      flow_1d,
                           const double      pressure_bc,
                           const VesselData &vessel_data) const;

  // Python: modelBloodFlow.py::model.inflowAorta (line 743)
  double
  compute_inflow_function(const double time, const int flow_type) const;

  // Python: modelBloodFlow.py::model.getkfromc0 (line 736)
  double
  compute_elastic_modulus_from_wave_speed(const double wave_speed,
                                          const double reference_area) const;

  // Python: modelBloodFlow.py::model.aFp (line 719) - inverse of pFa
  double
  compute_area_from_pressure(const double      pressure,
                             const VesselData &vessel_data) const;

  // Additional essential functions
  /**
   * @brief Compute pressure derivative with respect to area
   * Python equivalent: modelBloodFlow.py::model.dpda (line 106)
   */
  double
  compute_pressure_derivative(const double      area,
                              const VesselData &vessel_data) const;

  /**
   * @brief Compute eigenvalue matrix (diagonal matrix of eigenvalues)
   * Python equivalent: modelBloodFlow.py::model.lambdaMatrix (line 219)
   */
  FullMatrix<double>
  compute_eigenvalue_matrix(const Vector<double> &state,
                            const VesselData     &vessel_data) const;

  /**
   * @brief Set initial conditions for vessel
   * Python equivalent: vessel.py::vessel.setInitialCondition (line 98)
   */
  void
  set_initial_conditions(const double initial_pressure,
                         const double initial_velocity);

  /**
   * @brief Prescribe RCR boundary condition
   * Python equivalent: modelBloodFlow.py::model.prescribeRCR (referenced in
   * vessel.py:203)
   */
  Vector<double>
  prescribe_rcr_boundary_condition(const Vector<double> &interior_state,
                                   const double          rcr_pressure,
                                   const double          rcr_resistance,
                                   const VesselData     &vessel_data) const;

  /**
   * @brief Set boundary conditions for vessel
   * Python equivalent: vessel.py::vessel.setBoundaryConditions (line 188)
   */
  void
  set_boundary_conditions_for_vessel(const unsigned int vessel_id,
                                     const double       time);

  /**
   * @brief Update solution for one time step
   * Python equivalent: numerics.py::numericalMethod.update (line 208)
   */
  void
  update_solution_time_step();

  /**
   * @brief Apply ENO reconstruction (placeholder)
   * Python equivalent: numerics.py::numericalMethod.reconstructionENO (line
   * 302)
   */
  void
  apply_eno_reconstruction();

  /**
   * @brief Apply MUSCL reconstruction (placeholder)
   * Python equivalent: Used in vessel.py::vessel.getMusclHancockEvolveState
   * (line 170)
   */
  void
  apply_muscl_reconstruction();

  /**
   * @brief Compute source term (friction)
   * Python equivalent: modelBloodFlow.py::model.source (referenced in
   * vessel.py)
   */
  Vector<double>
  compute_source_term(const Vector<double> &state,
                      const VesselData     &vessel_data,
                      const double          time) const;

  /**
   * @brief Compute Lax-Wendroff numerical flux
   * Python equivalent: numerics.py::numericalMethod.numFluxLW (line 97)
   */
  Vector<double>
  compute_lax_wendroff_flux(const Vector<double> &state_left,
                            const Vector<double> &state_right,
                            const VesselData     &vessel_data,
                            const double          dx,
                            const double          dt) const;

  /**
   * @brief Compute FORCE numerical flux
   * Python equivalent: numerics.py::numericalMethod.numFluxFORCE (line 108)
   */
  Vector<double>
  compute_force_flux(const Vector<double> &state_left,
                     const Vector<double> &state_right,
                     const VesselData     &vessel_data,
                     const double          dx,
                     const double          dt) const;

  /**
   * @brief Evolve RCR boundary condition
   * Python equivalent: vessel.py::vessel.evolveRCR (line 155)
   */
  void
  evolve_rcr_boundary_condition(const double flow_rate,
                                const double time_step,
                                double      &rcr_pressure,
                                const double rcr_capacitance,
                                const double rcr_distal_resistance,
                                const double rcr_distal_pressure) const;

  /**
   * @brief Estimate wave speeds for HLL solver
   * Python equivalent: Used in numerics.py::numericalMethod.numFluxHLL (line
   * 145)
   */
  std::pair<double, double>
  estimate_wave_speeds_for_hll(const Vector<double> &state_left,
                               const Vector<double> &state_right,
                               const VesselData     &vessel_data) const;

  /**
   * @brief Get cell average state for boundary conditions
   * Python equivalent: vessel.py::vessel.getSolution (line 115)
   */
  Vector<double>
  get_cell_average_state(
    const typename DoFHandler<1, spacedim>::active_cell_iterator &cell) const;

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
  // Python: vessel.py::vessel instances stored in array
  std::vector<VesselData>                         vessel_data;
  std::map<types::global_dof_index, unsigned int> cell_to_vessel_map;

  // Face connectivity map for DG integration
  std::map<typename DoFHandler<1, spacedim>::face_iterator,
           std::vector<typename DoFHandler<1, spacedim>::active_cell_iterator>>
    face_to_cells_map;

  // Solution vectors
  // Python: vessel.py::vessel.Q (line 30) - state variables [area, flow]
  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> old_solution;
  LinearAlgebra::distributed::Vector<double> system_rhs;
  // Python: vessel.py::vessel.P (line 46) - pressure for output
  LinearAlgebra::distributed::Vector<double> pressure; // For output
  LinearAlgebra::distributed::Vector<double> velocity; // For output

  // Time stepping
  // Python: vessel.py::vessel.time (line 45)
  double current_time;
  // Python: vessel.py::vessel.dt (line 33)
  double       time_step;
  unsigned int timestep_number;

  // Output
  mutable unsigned int output_number;
};

#endif
