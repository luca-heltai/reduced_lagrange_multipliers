// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef immersx_coral_local_co_refinement_h
#define immersx_coral_local_co_refinement_h

#include <coral.h>
#include <immersx/grid/local_co_refinement.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace ImmersX::Coral
{
  template <int d1, int d2, int spacedim>
  inline void
  register_local_co_refinement_operation()
  {
    using Space      = std::shared_ptr<ImmersX::Domain<d1, spacedim>>;
    using Embedded   = std::shared_ptr<ImmersX::Domain<d2, spacedim>>;
    using Parameters = ImmersX::RefinementParameters<d2>;
    using Report     = ImmersX::LocalCoRefinementReport;
    coral::NodeObject::register_function(
      std::function<Report(Space &, Embedded &, const Parameters &)>(
        [](Space &space, Embedded &embedded, const Parameters &parameters) {
          AssertThrow(space && embedded,
                      dealii::ExcMessage(
                        "Co-refinement requires two non-null domains."));
          return ImmersX::LocalCoRefinement<d1, d2, spacedim>(parameters)
            .run(*space, *embedded);
        }),
      {"space", "embedded", "parameters"},
      method_metadata(
        "Co-refine domains",
        "Co-refine domains",
        "Space " + std::to_string(d1) + "D, embedded " + std::to_string(d2) +
          "D in " + std::to_string(spacedim) + "D",
        "Refine two distinct, unfinalized domains, retain imported fields, "
        "and finalize both meshes before constructing application FE spaces. "
        "Outputs are space, embedded, and the global violation report.",
        "LocalCoRefinement",
        "run"));
  }

  template <int d2, int spacedim>
  inline void
  register_local_co_refinement_space_dimensions()
  {
    register_local_co_refinement_operation<1, d2, spacedim>();
    if constexpr (spacedim >= 2)
      register_local_co_refinement_operation<2, d2, spacedim>();
    if constexpr (spacedim >= 3)
      register_local_co_refinement_operation<3, d2, spacedim>();
  }

  template <int dim, int spacedim>
  inline void
  register_local_co_refinement_embedded_dimension()
  {
    register_parameter_type<ImmersX::RefinementParameters<dim>>(
      "ImmersX::RefinementParameters<" + std::to_string(dim) + ">");
    register_local_co_refinement_space_dimensions<dim, spacedim>();
  }

  template <int spacedim>
  inline void
  register_local_co_refinement_types()
  {
    using Report = ImmersX::LocalCoRefinementReport;
    coral::detail::set_type_alias<Report>("ImmersX::LocalCoRefinementReport");
    coral::NodeObject::register_output_type<Report>();
    coral::detail::set_type_alias<unsigned long long>("unsigned long long");
    coral::NodeObject::register_elementary_type<unsigned long long>();

    for (const auto &entry :
         {std::make_pair("intersecting pairs", &Report::intersecting_pairs),
          std::make_pair("lower violations", &Report::lower_violations),
          std::make_pair("upper violations", &Report::upper_violations)})
      coral::NodeObject::register_function(
        std::function<unsigned long long(const Report &)>(
          [member = entry.second](const Report &report) {
            return report.*member;
          }),
        {"report"},
        method_metadata("Co-refinement " + std::string(entry.first),
                        "Co-refinement " + std::string(entry.first),
                        "Global count",
                        "Read a global bounding-box pair count.",
                        "LocalCoRefinementReport",
                        entry.first));

    coral::NodeObject::register_function(
      std::function<bool(const Report &)>(
        [](const Report &report) { return report.satisfied(); }),
      {"report"},
      method_metadata(
        "Co-refinement satisfied",
        "Co-refinement satisfied",
        "Global report",
        "Check both violation counts; zero pairs also satisfy the bounds.",
        "LocalCoRefinementReport",
        "satisfied"));
    coral::NodeObject::register_function(
      std::function<void(const Report &)>([](const Report &report) {
        AssertThrow(report.satisfied(),
                    dealii::ExcMessage(
                      "Co-refinement diameter bounds are not satisfied."));
      }),
      {"report"},
      method_metadata(
        "Require satisfied co-refinement",
        "Require satisfied co-refinement",
        "Optional strict check",
        "Explicitly fail when the report contains remaining violations. "
        "Co-refinement itself remains nonfatal.",
        "LocalCoRefinementReport",
        "require_satisfied"));

    register_local_co_refinement_embedded_dimension<1, spacedim>();
    if constexpr (spacedim >= 2)
      register_local_co_refinement_embedded_dimension<2, spacedim>();
    if constexpr (spacedim >= 3)
      register_local_co_refinement_embedded_dimension<3, spacedim>();
  }
} // namespace ImmersX::Coral
#endif
