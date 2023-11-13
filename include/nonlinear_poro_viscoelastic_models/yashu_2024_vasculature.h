#include "nonlinear_poro_viscoelasticity.h"

namespace NonLinearPoroViscoElasticity
{
  //@sect4{Base class: Cube geometry and loading pattern}
  template <int dim>
  class Yashu2024Vasculature : public Solid<dim>
  {
  public:
    Yashu2024Vasculature(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}

    virtual ~Yashu2024Vasculature()
    {}

  private:
    virtual void
    make_grid() override
    {
      GridGenerator::hyper_cube(this->triangulation, 0.0, 1.0, true);

      typename Triangulation<dim>::active_cell_iterator
        cell = this->triangulation.begin_active(),
        endc = this->triangulation.end();
      for (; cell != endc; ++cell)
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            if (cell->face(face)->at_boundary() == true &&
                (cell->face(face)->boundary_id() == 0 ||
                 cell->face(face)->boundary_id() == 1 ||
                 cell->face(face)->boundary_id() == 2 ||
                 cell->face(face)->boundary_id() == 3))

              cell->face(face)->set_boundary_id(100);
        }

      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(100, 100);
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.5 * this->parameters.scale;
      tracked_vertices[0][1] = 0.5 * this->parameters.scale;
      tracked_vertices[0][2] = 1.0 * this->parameters.scale;

      tracked_vertices[1][0] = 0.5 * this->parameters.scale;
      tracked_vertices[1][1] = 0.5 * this->parameters.scale;
      tracked_vertices[1][2] = 0.5 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->z_displacement));

      Point<dim> fix_node(0.5 * this->parameters.scale,
                          0.5 * this->parameters.scale,
                          0.0);
      typename DoFHandler<dim>::active_cell_iterator
        cell = this->dof_handler_ref.begin_active(),
        endc = this->dof_handler_ref.end();
      for (; cell != endc; ++cell)
        for (unsigned int node = 0; node < GeometryInfo<dim>::vertices_per_cell;
             ++node)
          {
            if ((abs(cell->vertex(node)[2] - fix_node[2]) <
                 (1e-6 * this->parameters.scale)) &&
                (abs(cell->vertex(node)[0] - fix_node[0]) <
                 (1e-6 * this->parameters.scale)))
              constraints.add_line(cell->vertex_dof_index(node, 0));

            if ((abs(cell->vertex(node)[2] - fix_node[2]) <
                 (1e-6 * this->parameters.scale)) &&
                (abs(cell->vertex(node)[1] - fix_node[1]) <
                 (1e-6 * this->parameters.scale)))
              constraints.add_line(cell->vertex_dof_index(node, 1));
          }

      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double>  value = get_dirichlet_load(5, 2);
          FEValuesExtractors::Scalar direction;
          direction = this->z_displacement;

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(direction));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 5)
            {
              const double final_load   = this->parameters.load;
              const double current_time = this->time->get_current();
              const double final_time   = this->time->get_end();
              const double num_cycles   = 3.0;

              return final_load / 2.0 *
                     (1.0 -
                      std::sin(
                        numbers::PI *
                        (2.0 * num_cycles * current_time / final_time + 0.5))) *
                     N;
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 5;
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 5) && (direction == 2))
        {
          const double final_displ    = this->parameters.load;
          const double current_time   = this->time->get_current();
          const double final_time     = this->time->get_end();
          const double delta_time     = this->time->get_delta_t();
          const double num_cycles     = 3.0;
          double       current_displ  = 0.0;
          double       previous_displ = 0.0;

          if (this->parameters.num_cycle_sets == 1)
            {
              current_displ =
                final_displ / 2.0 *
                (1.0 - std::sin(
                         numbers::PI *
                         (2.0 * num_cycles * current_time / final_time + 0.5)));
              previous_displ =
                final_displ / 2.0 *
                (1.0 - std::sin(numbers::PI *
                                (2.0 * num_cycles *
                                   (current_time - delta_time) / final_time +
                                 0.5)));
            }
          else
            {
              if (current_time <= (final_time * 1.0 / 3.0))
                {
                  current_displ =
                    final_displ / 2.0 *
                    (1.0 -
                     std::sin(numbers::PI * (2.0 * num_cycles * current_time /
                                               (final_time * 1.0 / 3.0) +
                                             0.5)));
                  previous_displ =
                    final_displ / 2.0 *
                    (1.0 -
                     std::sin(numbers::PI *
                              (2.0 * num_cycles * (current_time - delta_time) /
                                 (final_time * 1.0 / 3.0) +
                               0.5)));
                }
              else
                {
                  current_displ =
                    final_displ *
                    (1.0 -
                     std::sin(numbers::PI * (2.0 * num_cycles * current_time /
                                               (final_time * 2.0 / 3.0) -
                                             (num_cycles - 0.5))));
                  previous_displ =
                    final_displ *
                    (1.0 -
                     std::sin(numbers::PI *
                              (2.0 * num_cycles * (current_time - delta_time) /
                                 (final_time * 2.0 / 3.0) -
                               (num_cycles - 0.5))));
                }
            }
          displ_incr[2] = current_displ - previous_displ;
        }
      return displ_incr;
    }
  };
} // namespace NonLinearPoroViscoElasticity