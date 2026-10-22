// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coupled_poisson_elasticity_h
#define immersx_coupled_poisson_elasticity_h

#include <immersx/coupling/particle_coupling.h>

#include <string>

namespace CoupledPoissonElasticity
{
  /** Radial normal about the cylinder axis parallel to z. */
  class CylinderSurface
  {
  public:
    using Point = dealii::Point<3>;

    explicit CylinderSurface(
      const Point       &center     = Point(0.5, 0.5, 0.),
      const std::string &subsection = "/Pressure traction/Particle coupling/")
      : center_(center)
      , particle_parameters_(subsection)
    {}

    dealii::Tensor<1, 3>
    normal(const Point &point) const
    {
      dealii::Tensor<1, 3> radial;
      radial[0] = point[0] - center_[0];
      radial[1] = point[1] - center_[1];
      return radial / radial.norm();
    }

    const ImmersX::ParticleCouplingParameters<3> &
    particle_coupling_parameters() const
    {
      return particle_parameters_;
    }

  private:
    Point                                  center_;
    ImmersX::ParticleCouplingParameters<3> particle_parameters_;
  };
} // namespace CoupledPoissonElasticity

#endif // immersx_coupled_poisson_elasticity_h
