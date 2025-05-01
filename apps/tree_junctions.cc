#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "utils.h"

void
output_double_number(double input, const std::string &text)
{
  std::cout << text << input << std::endl;
}

class TestFunction : public Function<3>
{
public:
  TestFunction()
    : Function<3>()
  {}

  virtual double
  value(const Point<3> &p, const unsigned int component = 0) const override
  {
    (void)component;
    return p[0] + p[1] + p[2];
  }
};


// Helper point comparator
template <int dim>
struct PointComparator
{
  PointComparator(double tol = 1e-10)
    : tolerance(tol)
  {}

  bool
  operator()(const Point<dim> &p1, const Point<dim> &p2) const
  {
    for (unsigned int d = 0; d < dim; ++d)
      {
        if (std::abs(p1[d] - p2[d]) > tolerance)
          return p1[d] < p2[d];
      }
    return false; // Points are considered equal
  }

  double tolerance;
};


template <int dim, int spacedim>
void
test_interpolation_at_junctions_distributed(
  const DoFHandler<dim, spacedim> &dof_handler,
  double                           tolerance = 1e-10)
{
  static_assert(dim == 1 && spacedim == 3,
                "This function only works for 1D networks in 3D space.");

  const auto                         &tria = dof_handler.get_triangulation();
  const FiniteElement<dim, spacedim> &fe   = dof_handler.get_fe();
  const MPI_Comm                      comm = tria.get_communicator();
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);

  if (my_rank == 0)
    {
      std::cout
        << "========== INTERPOLATION TEST AT JUNCTIONS (DISTRIBUTED) =========="
        << std::endl;
      std::cout << "Using " << fe.get_name() << " elements on " << n_ranks
                << " MPI ranks" << std::endl;
    }

  // Get non-manifold faces local to this process
  const auto         junction_faces = GridUtils::get_non_manifold_faces(tria);
  const unsigned int n_local_junctions = junction_faces.size();
  const unsigned int n_total_junctions =
    Utilities::MPI::sum(n_local_junctions, comm);

  if (my_rank == 0)
    std::cout << "Found " << n_total_junctions
              << " junctions across all processes" << std::endl;

  if (n_total_junctions == 0)
    {
      if (my_rank == 0)
        std::cout << "No junctions found. Return." << std::endl;
      return;
    }

  TestFunction                               func;
  LinearAlgebra::distributed::Vector<double> interpolated;

  // Initialize vector with locally relevant DoFs (owned + ghosts) for
  // interpolation
  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  interpolated.reinit(locally_owned_dofs, locally_relevant_dofs, comm);

  VectorTools::interpolate(dof_handler, func, interpolated);
  interpolated.update_ghost_values();

  // Collect junction data on this process
  std::vector<GridUtils::JunctionInfo>
    local_junction_data; // this should have the size of the number of local
                         // junctions

  for (const auto &[face, insisting_cells] : junction_faces)
    {
      GridUtils::JunctionInfo junction;
      junction.point = face->center();

      for (const auto &cell : insisting_cells)
        {
          if (cell->is_locally_owned())
            {
              bool         vertex0_at_junction = (cell->face(0) == face);
              unsigned int junction_vertex_idx = vertex0_at_junction ? 0 : 1;

              typename DoFHandler<dim, spacedim>::active_cell_iterator dof_cell(
                &tria, cell->level(), cell->index(), &dof_handler);

              std::vector<types::global_dof_index> dof_indices(
                fe.dofs_per_cell);
              dof_cell->get_dof_indices(dof_indices);

              for (unsigned int j = 0; j < fe.dofs_per_vertex; ++j)
                {
                  const unsigned int dof_idx_in_cell =
                    fe.dofs_per_vertex * junction_vertex_idx + j;
                  junction.dof_indices.push_back(dof_indices[dof_idx_in_cell]);
                  junction.dof_values.push_back(
                    interpolated[dof_indices[dof_idx_in_cell]]);
                }
            }
        }

      if (!junction.dof_indices.empty())
        local_junction_data.push_back(junction);
    }

  // Gather all junction data to rank 0.
  // this vector will store junctions from all ranks
  std::vector<GridUtils::JunctionInfo> all_junction_data;

  if (n_ranks > 1)
    {
      // Pack local data
      std::vector<char> local_buffer;
      for (const auto &junction : local_junction_data)
        junction.pack(local_buffer);

      // Get buffer sizes from all ranks
      unsigned int     local_buffer_size = local_buffer.size();
      std::vector<int> buffer_sizes(n_ranks);
      MPI_Gather(&local_buffer_size,
                 1,
                 MPI_INT,
                 buffer_sizes.data(),
                 1,
                 MPI_INT,
                 0,
                 comm);

      // Gather all buffers to rank 0
      std::vector<char> global_buffer;
      if (my_rank == 0)
        {
          // compute displacements for each rank, we have a variable lengths
          // in each message...
          std::vector<int> displacements(n_ranks, 0);
          for (unsigned int i = 1; i < n_ranks; ++i)
            displacements[i] = displacements[i - 1] + buffer_sizes[i - 1];

          global_buffer.resize(displacements.back() + buffer_sizes.back());

          MPI_Gatherv(local_buffer.data(),
                      local_buffer.size(),
                      MPI_CHAR,
                      global_buffer.data(),
                      buffer_sizes.data(),
                      displacements.data(),
                      MPI_CHAR,
                      0,
                      comm);

          // Unpack global data
          std::size_t pos = 0;
          while (pos < global_buffer.size())
            {
              GridUtils::JunctionInfo junction;
              junction.unpack(global_buffer, pos);
              all_junction_data.push_back(junction);
            }
        }
      else
        {
          MPI_Gatherv(local_buffer.data(),
                      local_buffer.size(),
                      MPI_CHAR,
                      nullptr,
                      nullptr,
                      nullptr,
                      MPI_CHAR,
                      0,
                      comm);
        }
    }
  else
    {
      // only 1 process
      all_junction_data = local_junction_data;
    }

  // Only rank 0 analyzes the gathered data
  if (my_rank == 0)
    {
      unsigned int continuity_issues = 0;
      unsigned int accuracy_issues   = 0;

      // Group junctions by geometric location (they might be duplicated across
      // processes).
      // map[Point] = vector of junctions at this point
      std::
        map<Point<3>, std::vector<GridUtils::JunctionInfo>, PointComparator<3>>
                         junction_groups;
      PointComparator<3> point_comp(
        1e-10); // Use small tolerance for point comparison. [TODO] @fdrmrc: use
                // relative tolerance

      for (const auto &junction : all_junction_data)
        junction_groups[junction.point].push_back(junction);

      std::cout << "After grouping by location: " << junction_groups.size()
                << " unique junctions" << std::endl;

      // Analyze each junction
      for (const auto &[point, junctions] : junction_groups)
        {
          std::cout << "Junction at " << point << std::endl;

          // Combine all DoF data from this junction
          std::map<types::global_dof_index, double> all_dofs;
          for (const auto &junction : junctions)
            {
              for (std::size_t i = 0; i < junction.dof_indices.size(); ++i)
                all_dofs[junction.dof_indices[i]] = junction.dof_values[i];
            }

          // Check for continuity
          std::set<double> unique_values;
          for (const auto &[dof, value] : all_dofs)
            unique_values.insert(value);

          std::cout << "  DoF values at junction: ";
          for (double val : unique_values)
            std::cout << val << " ";
          std::cout << std::endl;

          // Check continuity
          if (unique_values.size() > 1)
            {
              std::cout << "  CONTINUITY ERROR: Multiple values at junction!"
                        << std::endl;
              continuity_issues++;
            }
          else
            {
              std::cout << "  Continuity check: PASSED" << std::endl;
            }

          // Check accuracy
          const double expected_value = func.value(point);
          const double actual_value =
            *unique_values.begin(); // we have only one value if things are ok
          const double error = std::abs(actual_value - expected_value);

          std::cout << "  Expected value: " << expected_value
                    << ", Actual value: " << actual_value
                    << ", Error: " << error << std::endl;

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

      // Summary
      std::cout << "  Continuity issues: " << continuity_issues << " out of "
                << junction_groups.size() << " junctions" << std::endl;
      std::cout << "  Accuracy issues: " << accuracy_issues << " out of "
                << junction_groups.size() << " junctions" << std::endl;

      if (continuity_issues == 0 && accuracy_issues == 0)
        std::cout << "  All tests PASSED" << std::endl;
      else
        std::cout << "  Some tests FAILED" << std::endl;
    }

  MPI_Barrier(comm);

  if (my_rank == 0)
    std::cout << "===================================================="
              << std::endl;
}

template <int dim, int spacedim>
void
test_DoF_at_junctions_distributed(const DoFHandler<dim, spacedim> &dof_handler)
{
  const auto                         &tria = dof_handler.get_triangulation();
  const FiniteElement<dim, spacedim> &fe   = dof_handler.get_fe();
  const MPI_Comm                      comm = tria.get_communicator();
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);

  if (my_rank == 0)
    std::cout << "\nDOF DISTRIBUTION AT JUNCTIONS (DISTRIBUTED):" << std::endl;

  // Get junction faces local to this process
  const auto         junction_faces = GridUtils::get_non_manifold_faces(tria);
  const unsigned int n_local_junctions = junction_faces.size();
  const unsigned int n_total_junctions =
    Utilities::MPI::sum(n_local_junctions, comm);

  if (my_rank == 0)
    std::cout << "Found " << n_total_junctions
              << " junctions across all processes" << std::endl;

  if (n_total_junctions == 0)
    {
      if (my_rank == 0)
        std::cout << "No junctions found. Return." << std::endl;
      return;
    }


  // Collect junction DoF data on this process
  std::vector<GridUtils::JunctionDoFData> local_junction_data;

  for (const auto &[face, cells] : junction_faces)
    {
      GridUtils::JunctionDoFData junction;
      junction.point = face->center();

      for (const auto &cell : cells)
        {
          if (cell->is_locally_owned())
            {
              bool         vertex0_at_junction = (cell->face(0) == face);
              unsigned int junction_vertex_idx = vertex0_at_junction ? 0 : 1;

              typename DoFHandler<dim, spacedim>::active_cell_iterator dof_cell(
                &tria, cell->level(), cell->index(), &dof_handler);

              std::vector<types::global_dof_index> dof_indices(
                fe.dofs_per_cell);
              dof_cell->get_dof_indices(dof_indices);

              for (unsigned int j = 0; j < fe.dofs_per_vertex; ++j)
                {
                  const unsigned int dof_idx_in_cell =
                    fe.dofs_per_vertex * junction_vertex_idx + j;
                  junction.dof_usage[dof_indices[dof_idx_in_cell]]++;
                }
            }
        }

      if (!junction.dof_usage.empty())
        local_junction_data.push_back(junction);
    }

  // Gather all junction data to rank 0
  std::vector<GridUtils::JunctionDoFData> all_junction_data;

  if (n_ranks > 1)
    {
      // Pack local data
      std::vector<char> local_buffer;
      for (const auto &junction : local_junction_data)
        junction.pack(local_buffer);

      // Get buffer sizes from all ranks
      unsigned int     local_buffer_size = local_buffer.size();
      std::vector<int> buffer_sizes(n_ranks);
      MPI_Gather(&local_buffer_size,
                 1,
                 MPI_INT,
                 buffer_sizes.data(),
                 1,
                 MPI_INT,
                 0,
                 comm);

      // Gather all buffers to rank 0
      std::vector<char> global_buffer;
      if (my_rank == 0)
        {
          std::vector<int> displacements(n_ranks, 0);
          for (unsigned int i = 1; i < n_ranks; ++i)
            displacements[i] = displacements[i - 1] + buffer_sizes[i - 1];

          global_buffer.resize(displacements.back() + buffer_sizes.back());

          MPI_Gatherv(local_buffer.data(),
                      local_buffer.size(),
                      MPI_CHAR,
                      global_buffer.data(),
                      buffer_sizes.data(),
                      displacements.data(),
                      MPI_CHAR,
                      0,
                      comm);

          // Unpack global data
          std::size_t pos = 0;
          while (pos < global_buffer.size())
            {
              GridUtils::JunctionDoFData junction;
              junction.unpack(global_buffer, pos);
              all_junction_data.push_back(junction);
            }
        }
      else
        {
          MPI_Gatherv(local_buffer.data(),
                      local_buffer.size(),
                      MPI_CHAR,
                      nullptr,
                      nullptr,
                      nullptr,
                      MPI_CHAR,
                      0,
                      comm);
        }
    }
  else
    {
      // Single process case
      all_junction_data = local_junction_data;
    }

  // Only rank 0 analyzes the gathered data
  if (my_rank == 0)
    {
      // Group junctions by geometric location
      std::map<Point<3>,
               std::vector<GridUtils::JunctionDoFData>,
               PointComparator<3>>
                         junction_groups;
      PointComparator<3> point_comp(1e-8);

      for (const auto &junction : all_junction_data)
        junction_groups[junction.point].push_back(junction);

      std::cout << "After grouping by location: " << junction_groups.size()
                << " unique junctions" << std::endl;

      // Analyze each junction
      for (const auto &[point, junctions] : junction_groups)
        {
          std::cout << "Junction at " << point << ":" << std::endl;

          // Combine DoF usage from all processes
          std::map<types::global_dof_index, unsigned int> combined_dof_usage;
          std::set<types::global_dof_index>               all_dofs;

          for (const auto &junction : junctions)
            {
              for (const auto &[dof_idx, count] : junction.dof_usage)
                {
                  combined_dof_usage[dof_idx] += count;
                  all_dofs.insert(dof_idx);
                }
            }

          // Analyze DoF sharing pattern
          std::map<unsigned int, unsigned int> dof_sharing_histogram;
          for (const auto &[dof_idx, count] : combined_dof_usage)
            dof_sharing_histogram[count]++;

          for (const auto &[count, frequency] : dof_sharing_histogram)
            {
              std::cout << "  " << frequency << " DoFs shared by " << count
                        << " cells" << std::endl;
            }

          // Show the actual DoF indices
          std::cout << "  DoF indices at junction:" << std::endl;
          for (const auto &[dof_idx, count] : combined_dof_usage)
            std::cout << "    DoF " << dof_idx << " used by " << count
                      << " cells" << std::endl;
        }
    }
}


template <int dim, int spacedim>
void
test_distributed()
{
  MPI_Comm           mpi_comm = MPI_COMM_WORLD;
  const unsigned int my_rank  = Utilities::MPI::this_mpi_process(mpi_comm);
  const unsigned int n_ranks  = Utilities::MPI::n_mpi_processes(mpi_comm);

  if (my_rank == 0)
    {
      std::cout << "Running distributed test with " << n_ranks << " MPI ranks"
                << std::endl;
      std::cout << "Reading network mesh from VTK file..." << std::endl;
    }

  // First, create serial triangulation to read the file
  Triangulation<dim, spacedim> serial_tria;

  const std::string     filename = SOURCE_DIR "/data/tests/mstree_100.vtk";
  GridIn<dim, spacedim> gridin;
  gridin.attach_triangulation(serial_tria);
  std::ifstream input_file(filename);
  gridin.read_vtk(input_file);
  if (my_rank == 0)
    std::cout << "Serial mesh has " << serial_tria.n_active_cells() << " cells"
              << std::endl;

  // Partition serial triangulation:
  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(mpi_comm),
                                     serial_tria);

  // 4. Create construction data for fully distributed triangulation
  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, mpi_comm);

  // 5. Create fully distributed triangulation
  parallel::fullydistributed::Triangulation<dim, spacedim> distributed_tria(
    mpi_comm);
  distributed_tria.create_triangulation(construction_data);

  if (my_rank == 0)
    std::cout << "Created distributed triangulation" << std::endl;

  // Print cell distribution
  const unsigned int n_local_cells =
    distributed_tria.n_locally_owned_active_cells();
  const unsigned int n_global_cells =
    Utilities::MPI::sum(distributed_tria.n_locally_owned_active_cells(),
                        mpi_comm);

  std::cout << "Rank " << my_rank << " has " << n_local_cells
            << " local cells out of " << n_global_cells << " total"
            << std::endl;

  // Setup DoF handler
  DoFHandler<dim, spacedim> dof_handler(distributed_tria);
  FE_Q<dim, spacedim>       fe(1);
  dof_handler.distribute_dofs(fe);

  const unsigned int n_local_dofs = dof_handler.n_locally_owned_dofs();
  const unsigned int n_global_dofs =
    Utilities::MPI::sum(dof_handler.n_locally_owned_dofs(), mpi_comm);

  std::cout << "Rank " << my_rank << " has " << n_local_dofs
            << " local DoFs out of " << n_global_dofs << " total" << std::endl;

  // Run the tests
  test_interpolation_at_junctions_distributed<dim, spacedim>(dof_handler);
  test_DoF_at_junctions_distributed<dim, spacedim>(dof_handler);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test_distributed<1, 3>();

  return 0;
}
