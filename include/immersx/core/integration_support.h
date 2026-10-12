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

#include <deal.II/grid/tria.h>

#include <optional>

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
} // namespace ImmersX

#endif // immersx_integration_support_h
