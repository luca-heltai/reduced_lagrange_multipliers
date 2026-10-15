// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_integration_support_h
#define immersx_integration_support_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values_base.h>

#include <deal.II/grid/tria.h>

#include <optional>
#include <set>
#include <type_traits>

namespace ImmersX
{
  /** A non-owning view of the cells used by a weak-term integral. */
  template <int dim, int spacedim = dim>
  class IntegrationSupport
  {
  public:
    static constexpr int
    dimension() noexcept
    {
      return dim;
    }

    static constexpr int
    spacedimension() noexcept
    {
      return spacedim;
    }

    explicit IntegrationSupport(
      const dealii::Triangulation<dim, spacedim> &tria)
      : triangulation_(&tria)
    {}

    IntegrationSupport(const dealii::Triangulation<dim, spacedim> &tria,
                       const dealii::types::material_id            material_id)
      : triangulation_(&tria)
      , material_id_(material_id)
    {}

    const dealii::Triangulation<dim, spacedim> &
    triangulation() const
    {
      AssertThrow(triangulation_ != nullptr,
                  dealii::ExcMessage("Integration support has no mesh."));
      return *triangulation_;
    }

    bool
    has_material_filter() const
    {
      return material_id_.has_value();
    }

    std::optional<dealii::types::material_id>
    material_id() const
    {
      return material_id_;
    }

    template <typename CellIterator>
    bool
    includes_cell(const CellIterator &cell) const
    {
      return !material_id_ || cell->material_id() == *material_id_;
    }

    template <typename TriangulationType>
    void
    assert_matches(const TriangulationType &tria) const
    {
      AssertThrow(static_cast<const void *>(triangulation_) ==
                    static_cast<const void *>(&tria),
                  dealii::ExcMessage(
                    "The integration support and weak-term fields must use "
                    "the same triangulation."));
    }

  private:
    const dealii::Triangulation<dim, spacedim> *triangulation_ = nullptr;
    std::optional<dealii::types::material_id>   material_id_;
  };

  /** Marker used when a weak term infers its historical integration region. */
  struct InferredIntegrationSupport
  {};

  template <typename Type>
  struct is_integration_support : std::false_type
  {};

  template <int dim, int spacedim>
  struct is_integration_support<IntegrationSupport<dim, spacedim>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct BoundaryNormalDirection
  {
    template <typename FaceValues>
    dealii::Tensor<1, spacedim>
    evaluate(const FaceValues &values, const unsigned int q) const
    {
      static_assert(dim == spacedim,
                    "Use conormal() for the outward direction of an "
                    "embedded-domain boundary.");
      return values.normal_vector(q);
    }
  };

  template <int dim, int spacedim>
  struct BoundaryConormalDirection
  {
    template <typename FaceValues>
    dealii::Tensor<1, spacedim>
    evaluate(const FaceValues &values, const unsigned int q) const
    {
      static_assert(dim < spacedim,
                    "conormal() is defined here for embedded-domain "
                    "boundaries.");
      return values.normal_vector(q);
    }
  };

  template <int dim, int spacedim>
  struct BoundaryTangentDirection
  {
    template <typename FaceValues>
    dealii::Tensor<1, spacedim>
    evaluate(const FaceValues &values, const unsigned int q) const
    {
      static_assert(dim == 2 && spacedim == 2,
                    "A unique boundary tangent is available only for a "
                    "planar curve boundary in this support.");
      const auto           normal = values.normal_vector(q);
      dealii::Tensor<1, 2> tangent;
      tangent[0] = normal[1];
      tangent[1] = -normal[0];
      return tangent;
    }
  };

  /** A non-owning view of boundary faces selected by boundary id. */
  template <int dim, int spacedim = dim>
  class BoundaryIntegrationSupport
  {
  public:
    static constexpr int
    dimension() noexcept
    {
      return dim;
    }

    static constexpr int
    spacedimension() noexcept
    {
      return spacedim;
    }

    explicit BoundaryIntegrationSupport(
      const dealii::Triangulation<dim, spacedim> &tria)
      : triangulation_(&tria)
    {
      const auto boundary_ids = tria.get_boundary_ids();
      boundary_ids_.insert(boundary_ids.begin(), boundary_ids.end());
    }

    BoundaryIntegrationSupport(const dealii::Triangulation<dim, spacedim> &tria,
                               const dealii::types::boundary_id boundary_id)
      : triangulation_(&tria)
      , boundary_ids_{boundary_id}
    {}

    const dealii::Triangulation<dim, spacedim> &
    triangulation() const
    {
      AssertThrow(triangulation_ != nullptr,
                  dealii::ExcMessage("Boundary support has no mesh."));
      return *triangulation_;
    }

    const std::set<dealii::types::boundary_id> &
    boundary_ids() const
    {
      return boundary_ids_;
    }

    template <int d = dim, std::enable_if_t<d == spacedim, int> = 0>
    BoundaryNormalDirection<dim, spacedim>
    normal() const
    {
      return {};
    }

    template <int d = dim,
              std::enable_if_t<d<spacedim, int> = 0>
                BoundaryConormalDirection<dim, spacedim> conormal() const
    {
      return {};
    }

    template <int d                                   = dim,
              int s                                   = spacedim,
              std::enable_if_t<d == 2 && s == 2, int> = 0>
    BoundaryTangentDirection<dim, spacedim>
    tangent() const
    {
      return {};
    }

    template <typename TriangulationType>
    void
    assert_matches(const TriangulationType &tria) const
    {
      AssertThrow(static_cast<const void *>(triangulation_) ==
                    static_cast<const void *>(&tria),
                  dealii::ExcMessage(
                    "The boundary support and weak-term fields must use "
                    "the same triangulation."));
    }

  private:
    const dealii::Triangulation<dim, spacedim> *triangulation_ = nullptr;
    std::set<dealii::types::boundary_id>        boundary_ids_;
  };
} // namespace ImmersX

#endif // immersx_integration_support_h
