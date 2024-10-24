/* ---------------------------------------------------------------------
 */
#ifndef dealii_distributed_lagrange_multiplier_elasticity_h
#define dealii_distributed_lagrange_multiplier_elasticity_h

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// #include <deal.II/trilinos/parameter_acceptor.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/meshworker/simple.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/matrix_tools.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>

#include "inclusions.h"


#ifdef DEAL_II_WITH_OPENCASCADE
#  include <TopoDS.hxx>
#endif
#include <deal.II/base/hdf5.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>


template <int dim, int spacedim = dim>
class ElasticityProblemParameters : public ParameterAcceptor
{
public:
  ElasticityProblemParameters();

  std::string                   output_directory   = ".";
  std::string                   output_name        = "solution";
  unsigned int                  fe_degree          = 1;
  unsigned int                  initial_refinement = 5;
  std::list<types::boundary_id> dirichlet_ids{0};
  std::list<types::boundary_id> neumann_ids{};
  std::set<types::boundary_id>  normal_flux_ids{};
  std::string                   domain_type         = "generate";
  std::string                   name_of_grid        = "hyper_cube";
  std::string                   arguments_for_grid  = "-1: 1: false";
  std::string                   refinement_strategy = "fixed_fraction";
  double                        coarsening_fraction = 0.0;
  double                        refinement_fraction = 0.3;
  unsigned int                  n_refinement_cycles = 1;
  unsigned int                  max_cells           = 20000;
  bool                          output_pressure     = false;
  bool                          pressure_coupling   = false;
  double                        penalty_term        = 1.0e4;
  double                        wave_ampltiude      = 0.01;

  double Lame_mu            = 1;
  double Lame_lambda        = 1;
  double lambda_CSF         = 1;
  double mu_CSF             = 1;
  double lambda_Thalamus    = 1;
  double mu_Thalamus        = 1;
  double lambda_HPC         = 1;
  double mu_HPC             = 1;
  double lambda_WM          = 1;
  double mu_WM              = 1;
  double lambda_CC          = 1;
  double mu_CC              = 1;
  double lambda_Cerebellum  = 1;
  double mu_Cerebellum      = 1;
  double lambda_Cortex      = 1;
  double mu_Cortex          = 1;
  double lambda_BS          = 1;
  double mu_BS              = 1;
  double lambda_BG          = 1;
  double mu_BG              = 1;
  double lambda_Amygdala    = 1;
  double mu_Amygdala        = 1;
  double rho                = 1;
  double neta               = 1;
  double elasticity_modulus = 1;
  double relaxation_time    = 1;
  bool   linear_elasticity  = false;
  bool   rayleigh_damping   = false;
  double alpha_ray          = 0.1;
  double beta_ray           = 0.01;
  bool   kelvin_voigt       = false;
  bool   maxwell            = false;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    exact_solution;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> bc;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    Neumann_bc;

  std::string weight_expression = "1.";

  mutable ParameterAcceptorProxy<ReductionControl> inner_control;
  mutable ParameterAcceptorProxy<ReductionControl> outer_control;

  bool output_results_before_solving = false;

  mutable ParsedConvergenceTable convergence_table;

  // Time dependency.
  double initial_time = 0.0;
  double final_time   = 0.0;
  double dt           = 5e-3;
  double beta         = 0.25;
  double gamma        = 0.5;
};



template <int dim, int spacedim>
ElasticityProblemParameters<dim, spacedim>::ElasticityProblemParameters()
  : ParameterAcceptor("/Immersed Problem/")
  , rhs("/Immersed Problem/Right hand side", spacedim)
  , exact_solution("/Immersed Problem/Exact solution", spacedim)
  , bc("/Immersed Problem/Dirichlet boundary conditions", spacedim)
  , Neumann_bc("/Immersed Problem/Neumann boundary conditions", spacedim)
  , inner_control("/Immersed Problem/Solver/Inner control")
  , outer_control("/Immersed Problem/Solver/Outer control")
  , convergence_table(std::vector<std::string>(spacedim, "u"))
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("Output results also before solving",
                output_results_before_solving);
  add_parameter("Initial refinement", initial_refinement);
  add_parameter("Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Neumann boundary ids", neumann_ids);
  add_parameter("Normal flux boundary ids", normal_flux_ids);
  add_parameter("Output pressure", output_pressure);
  add_parameter(
    "Pressure coupling",
    pressure_coupling,
    "If this is true, then we do NOT solve a saddle point problem, but we use the "
    "input data as a pressure field on the vasculature network, and we solve for "
    "the displacement field directly.");
  enter_subsection("Grid generation");
  {
    add_parameter("Domain type",
                  domain_type,
                  "",
                  this->prm,
                  Patterns::Selection("generate|file|cheese|cylinder"));
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
  enter_subsection("Physical constants");
  {
    add_parameter("density", rho);
    enter_subsection("Weak Boundary");
    {
      add_parameter("Penalty term", penalty_term);
      add_parameter("Wave amplitude", wave_ampltiude);
    }
    leave_subsection();
    enter_subsection("Linear elasticity");
    {
      add_parameter("linear elasticity", linear_elasticity);
      add_parameter("Lame mu", Lame_mu);
      add_parameter("Lame lambda", Lame_lambda);
      add_parameter("CSF lambda", lambda_CSF);
      add_parameter("CSF mu", mu_CSF);
      add_parameter("Thalamus lambda", lambda_Thalamus);
      add_parameter("Thalamus mu", mu_Thalamus);
      add_parameter("HPC lambda", lambda_HPC);
      add_parameter("HPC mu", mu_HPC);
      add_parameter("WM lambda", lambda_WM);
      add_parameter("WM mu", mu_WM);
      add_parameter("CC lambda", lambda_CC);
      add_parameter("CC mu", mu_CC);
      add_parameter("Cerebellum lambda", lambda_Cerebellum);
      add_parameter("Cerebellum mu", mu_Cerebellum);
      add_parameter("Cortex lambda", lambda_Cortex);
      add_parameter("Cortex mu", mu_Cortex);
      add_parameter("BS lambda", lambda_BS);
      add_parameter("BS mu", mu_BS);
      add_parameter("BG lambda", lambda_BG);
      add_parameter("BG mu", mu_BG);
      add_parameter("Amygdala lambda", lambda_Amygdala);
      add_parameter("Amygdala mu", mu_Amygdala);
    }
    leave_subsection();
    enter_subsection("Rayleigh damping");
    {
      add_parameter("rayleigh damping", rayleigh_damping);
      add_parameter("alpha", alpha_ray);
      add_parameter("beta", beta_ray);
    }
    leave_subsection();
    enter_subsection("Kelvin Voigt");
    {
      add_parameter("kelvin voigt", kelvin_voigt);
      add_parameter("viscocity", neta);
      add_parameter("elasticity modulus", elasticity_modulus);
    }
    leave_subsection();
    enter_subsection("Maxwell");
    {
      add_parameter("maxwell", maxwell);
      add_parameter("relaxation time", relaxation_time);
      add_parameter("elasticity modulus", elasticity_modulus);
    }
    leave_subsection();
  }
  leave_subsection();
  enter_subsection("Exact solution");
  {
    add_parameter("Weight expression", weight_expression);
  }
  leave_subsection();
  enter_subsection("Time dependency");
  {
    add_parameter("Initial time", initial_time);
    add_parameter("Final time", final_time);
    add_parameter("Time step", dt);
    add_parameter("beta", beta);
    add_parameter("gamma", gamma);
  }
  leave_subsection();

  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();

  auto reset_function = [this]() {
    this->prm.set("Function expression", (spacedim == 2 ? "0; 0" : "0; 0; 0"));
  };
  rhs.declare_parameters_call_back.connect(reset_function);
  exact_solution.declare_parameters_call_back.connect(reset_function);
  Neumann_bc.declare_parameters_call_back.connect(reset_function);
  bc.declare_parameters_call_back.connect(reset_function);
}



template <int dim, int spacedim = dim>
class ElasticityProblem : public Subscriptor
{
public:
  ElasticityProblem(const ElasticityProblemParameters<dim, spacedim> &par);
  void
  make_grid();
  void
  setup_fe();
  void
  setup_dofs();
  void
  assemble_elasticity_system();
  void
  assemble_coupling();
  void
  run();
  void
  check_boundary_ids();

  /**
   * Builds coupling sparsity, and returns locally relevant inclusion dofs.
   */
  IndexSet
  assemble_coupling_sparsity(DynamicSparsityPattern &dsp);

  void
  solve();

  void
  refine_and_transfer();

  std::string
  output_solution() const;

  void
  output_results() const;

  void
  print_parameters() const;

  void
  compute_boundary_stress(bool openfilefirsttime) const; // make const

  void
  output_pressure(bool openfilefirsttime) const;

  // protected:
  const ElasticityProblemParameters<dim, spacedim> &par;
  MPI_Comm                                          mpi_communicator;
  ConditionalOStream                                pcout;
  mutable TimerOutput                               computing_timer;
  parallel::distributed::Triangulation<spacedim>    tria;
  std::unique_ptr<FiniteElement<spacedim>>          fe;
  Inclusions<spacedim>                              inclusions;
  std::unique_ptr<Quadrature<spacedim>>             quadrature;
  std::unique_ptr<Quadrature<spacedim - 1>>         face_quadrature_formula;
  DoFHandler<spacedim>                              dh;
  std::vector<IndexSet>                             owned_dofs;
  std::vector<IndexSet>                             relevant_dofs;

  AffineConstraints<double> constraints;
  AffineConstraints<double> inclusion_constraints;
  AffineConstraints<double> mean_value_constraints;

  LA::MPI::SparseMatrix stiffness_matrix;
  LA::MPI::SparseMatrix mass_matrix;
  LA::MPI::SparseMatrix force_matrix;
  LA::MPI::SparseMatrix coupling_matrix;
  LA::MPI::SparseMatrix damping_term;

  LA::MPI::SparseMatrix                           inclusion_matrix;
  LA::MPI::BlockVector                            solution;
  LA::MPI::BlockVector                            velocity;
  LA::MPI::BlockVector                            acceleration;
  LA::MPI::BlockVector                            predictor;
  LA::MPI::BlockVector                            corrector;
  LA::MPI::BlockVector                            locally_relevant_solution;
  LA::MPI::BlockVector                            system_rhs;
  LA::MPI::BlockVector                            system_lhs;
  LA::MPI::BlockVector                            system_rhs_f;
  std::vector<std::vector<BoundingBox<spacedim>>> global_bounding_boxes;
  unsigned int                                    cycle = 0;

  FEValuesExtractors::Vector displacement;

  // Postprocessing values
  std::map<types::boundary_id, Tensor<1, spacedim>> forces;
  std::map<types::boundary_id, Tensor<1, spacedim>> average_displacements;
  std::map<types::boundary_id, Tensor<1, spacedim>> average_normals;
  std::map<types::boundary_id, double>              areas;
  // std::vector<BaseClass::BlockType>                 pressure_records;

  // Time dependency.
  double current_time = 0.0;

  // mutable std::unique_ptr<HDF5::File> pressure_file;
  // std::ofstream pressure_file;
  // std::ofstream forces_file;
};


#endif