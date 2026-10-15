// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_coral_lifted_poisson_h
#define immersx_coral_lifted_poisson_h
#include <immersx/coral/register_primitives.h>
#include <immersx/physics/lifted_poisson.h>

namespace ImmersX::Coral
{
  /** Fix factory keys before registration: default template argument spelling
   * otherwise depends on the compiler and on which application is included
   * first.
   */
  inline void
  register_lifted_factory_keys()
  {
    using Domain         = ImmersX::Domain<3, 3>;
    using DomainHandle   = std::shared_ptr<Domain>;
    using Representative = std::shared_ptr<ImmersX::Domain<1, 3>>;
    using Space          = ImmersX::FiniteElementSpace<3, 3>;
    using SpaceHandle    = std::shared_ptr<Space>;
    using View           = ImmersX::FiniteElementSpaceView<3, 3>;
    using Conditions     = ImmersX::BoundaryConditions<3, 3>;
    coral::detail::set_type_alias<
      std::function<DomainHandle(const ImmersX::DomainParameters<3, 3> &)>>(
      "ImmersX::CreateDomainFactory<3,3>()");
    coral::detail::set_type_alias<
      std::function<ImmersX::LocalCoRefinementReport(
        DomainHandle &,
        Representative &,
        const ImmersX::RefinementParameters<1> &)>>(
      "ImmersX::CoRefineDomainsFactory<3,1,3>()");
    coral::detail::set_type_alias<std::function<
      SpaceHandle(const DomainHandle &,
                  const ImmersX::FiniteElementSpaceParameters<3, 3> &)>>(
      "ImmersX::CreateSpaceFactory<3,3>()");
    coral::detail::set_type_alias<std::function<View(const SpaceHandle &)>>(
      "ImmersX::SpaceViewFactory<3,3>()");
    const auto boundary_keys = [](auto extractor, const std::string &kind) {
      using Field = ImmersX::Field<3, 3, decltype(extractor)>;
      coral::detail::set_type_alias<std::function<Conditions(const Field &)>>(
        "ImmersX::BoundaryConditionsFactory<3," + kind + ">()");
      coral::detail::set_type_alias<
        std::function<Field(const Conditions &, const Field &)>>(
        "ImmersX::ApplyBoundaryFactory<3," + kind + ">()");
    };
    boundary_keys(dealii::FEValuesExtractors::Scalar(0), "Scalar");
    boundary_keys(dealii::FEValuesExtractors::Vector(0), "Vector");
  }

  inline void
  register_lifted_poisson_types()
  {
    register_primitive_types<3, 3>();
    register_owned_finite_element_space_types<1, 3>();
    register_primitive_finite_element_space_types<1, 3>();
    using Bulk  = ImmersX::Field<3, 3, dealii::FEValuesExtractors::Scalar>;
    using Modal = ImmersX::Field<1, 3, dealii::FEValuesExtractors::Scalar>;
    using Lift  = ImmersX::TensorProductLift<1, 2, 3>;
    using NativeGeometry = ImmersX::ModalLiftedObservable<Modal>;
    using NativeInteraction =
      ImmersX::LiftedMultiplierInteraction<Bulk, NativeGeometry>;
    using Geometry      = std::shared_ptr<NativeGeometry>;
    using Interaction   = std::shared_ptr<NativeInteraction>;
    using Adapter       = ImmersX::LiftedPoissonAdapter;
    using AdapterHandle = std::shared_ptr<Adapter>;
    using State         = ImmersX::LiftedPoissonState;
    using Input         = ImmersX::ImportedFiniteElementFields<1, 3>::FieldView;
    using Function      = ImmersX::Function<3>;
    register_parameter_type<Lift>("ImmersX::TensorProductLift<1,2,3>");
    coral::detail::set_type_alias<Geometry>(
      "ImmersX::ModalLiftedObservable<1,2,3>");
    coral::detail::set_type_alias<Interaction>(
      "ImmersX::LiftedMultiplierInteraction<3>");
    coral::NodeObject::register_output_type<Geometry>();
    coral::NodeObject::register_output_type<Interaction>();

    using MakeLift = std::function<Geometry(const Modal &, const Lift &)>;
    coral::detail::set_type_alias<MakeLift>(
      "ImmersX::ModalLiftFactory<1,2,3>()");
    coral::NodeObject::register_function(
      MakeLift([](const Modal &field, const Lift &lift) {
        return std::make_shared<NativeGeometry>(
          ImmersX::modal_lift(field, lift));
      }),
      {"modal coefficients", "lift parameters"},
      method_metadata(
        "Modal lift",
        "Modal lift",
        "Spatial radius",
        "Lift every representative FE component with its selected transverse mode.",
        "ModalLift",
        "create"));
    using MakeImportedLift =
      std::function<Geometry(const Modal &, const Lift &, const Input &)>;
    coral::detail::set_type_alias<MakeImportedLift>(
      "ImmersX::ImportedModalLiftFactory<1,2,3>()");
    coral::NodeObject::register_function(
      MakeImportedLift(
        [](const Modal &field, const Lift &lift, const Input &radius) {
          return std::make_shared<NativeGeometry>(
            ImmersX::modal_lift(field,
                                lift,
                                ImmersX::modal_radius(ImmersX::frozen(
                                  radius.field(), radius.coefficients()))));
        }),
      {"modal coefficients", "lift parameters", "radius field"},
      method_metadata(
        "Modal lift",
        "Modal lift",
        "Imported radius",
        "Use a retained imported scalar radius with distributed FE coefficients.",
        "ModalLift",
        "create"));
    using MakeData = std::function<Function(const std::string &, const Lift &)>;
    coral::detail::set_type_alias<MakeData>("ImmersX::ModalDataFactory<3>()");
    coral::NodeObject::register_function(
      MakeData([](const std::string &expressions, const Lift &lift) {
        const ImmersX::TensorProductLiftSupport<1, 2, 3> resolved(
          lift.parameters());
        auto function = std::make_shared<dealii::FunctionParser<3>>(
          resolved.reference_cross_section().n_selected_basis());
        function->initialize("x,y,z,t",
                             expressions,
                             {{"pi", dealii::numbers::PI},
                              {"E", dealii::numbers::E}},
                             true);
        return Function{function};
      }),
      {"coefficient expressions", "lift parameters"},
      method_metadata(
        "Modal data",
        "Modal data",
        "Selected-mode coefficients",
        "Prescribe representative modal data, not a pointwise surface value.",
        "ModalLift",
        "data"));
    using Data         = ImmersX::ModalDataEvaluator;
    using DomainHandle = std::shared_ptr<ImmersX::Domain<1, 3>>;
    coral::detail::set_type_alias<Data>("ImmersX::ModalDataEvaluator<3>");
    coral::NodeObject::register_output_type<Data>();
    using ImportedDataFactory = std::function<Data(const DomainHandle &,
                                                   const std::string &,
                                                   const Lift &,
                                                   const std::string &)>;
    coral::detail::set_type_alias<ImportedDataFactory>(
      "ImmersX::ImportedModalDataFactory<3>()");
    coral::NodeObject::register_function(
      ImportedDataFactory([](const DomainHandle &domain,
                             const std::string  &expressions,
                             const Lift         &lift,
                             const std::string  &selector) {
        const ImmersX::TensorProductLiftSupport<1, 2, 3> support(
          lift.parameters());
        return ImmersX::modal_data_evaluator(*domain,
                                             expressions,
                                             support.selected_modes().size(),
                                             selector);
      }),
      {"representative domain",
       "coefficient expressions",
       "lift parameters",
       "input field selector"},
      method_metadata(
        "Modal data",
        "Modal data",
        "Imported coefficients",
        "Evaluate modal coefficients using Domain-owned imported scalar aliases.",
        "ModalLift",
        "data"));
    using ImportedInteractionFactory =
      std::function<Interaction(const Bulk &, const Geometry &, const Data &)>;
    coral::detail::set_type_alias<ImportedInteractionFactory>(
      "ImmersX::ImportedLiftedMultiplierFactory<3>()");
    coral::NodeObject::register_function(
      ImportedInteractionFactory(
        [](const Bulk &bulk, const Geometry &lift, const Data &data) {
          return std::make_shared<NativeInteraction>(bulk, *lift, data);
        }),
      {"bulk field", "modal observable", "modal data"},
      method_metadata(
        "Lifted multiplier interaction",
        "Lifted multiplier interaction",
        "Imported modal coefficients",
        "Build the section-weighted constraint using retained imported modal data.",
        "LiftedMultiplierInteraction",
        "create"));

    using MakeInteraction = std::function<
      Interaction(const Bulk &, const Geometry &, const Function &)>;
    coral::detail::set_type_alias<MakeInteraction>(
      "ImmersX::LiftedMultiplierFactory<3>()");
    coral::NodeObject::register_function(
      MakeInteraction(
        [](const Bulk &bulk, const Geometry &lift, const Function &data) {
          return std::make_shared<NativeInteraction>(bulk, *lift, data.value);
        }),
      {"bulk field", "modal observable", "modal data"},
      method_metadata(
        "Lifted multiplier interaction",
        "Lifted multiplier interaction",
        "Scalar modal constraint",
        "Build the modal constraint and its section-weighted AL metric.",
        "LiftedMultiplierInteraction",
        "create"));
    using AddInteraction = std::function<
      Modal(AdapterHandle &, const Interaction &, const std::string &)>;
    coral::detail::set_type_alias<AddInteraction>(
      "ImmersX::AddLiftedMultiplier<3>()");
    coral::NodeObject::register_function(
      AddInteraction([](AdapterHandle     &adapter,
                        const Interaction &interaction,
                        const std::string &prefix) {
        return adapter->add(*interaction, prefix).fields().multiplier;
      }),
      {"adapter", "interaction", "prefix"},
      method_metadata(
        "Add lifted multiplier",
        "Add lifted multiplier",
        "Linear execution",
        "Register the interaction directly in the adapter; return its semantic multiplier field.",
        "LinearExecution",
        "add_interaction"));
    using Coefficient = std::function<Modal(const Modal &, unsigned int)>;
    coral::detail::set_type_alias<Coefficient>(
      "ImmersX::ModalCoefficientField<1,3>()");
    coral::NodeObject::register_function(
      Coefficient([](const Modal &field, unsigned int component) {
        AssertThrow(component < field.finite_element().n_components(),
                    dealii::ExcMessage("Invalid modal component."));
        return field.space()
          .field("lambda_" + std::to_string(component),
                 dealii::FEValuesExtractors::Scalar(component))
          .with_id(field.id());
      }),
      {"multiplier field", "component"},
      method_metadata(
        "Modal output coefficient field",
        "Modal output coefficient field",
        "Representative output",
        "Select a coefficient for native FE output without allocating another state field.",
        "ModalLift",
        "coefficient_field"));
    using Output = std::function<void(const AdapterHandle &,
                                      const State &,
                                      const Bulk &,
                                      const Geometry &,
                                      const Modal &,
                                      const std::string &)>;
    coral::detail::set_type_alias<Output>("ImmersX::WriteModalLift<3>()");
    coral::NodeObject::register_function(
      Output([](const AdapterHandle &adapter,
                const State         &state,
                const Bulk          &bulk,
                const Geometry      &geometry,
                const Modal         &field,
                const std::string   &filename) {
        geometry->write_lifted_output(bulk,
                                      adapter->field(state, field),
                                      filename);
      }),
      {"adapter",
       "state",
       "bulk field",
       "modal observable",
       "multiplier field",
       "filename"},
      method_metadata(
        "Write modal lift",
        "Write modal lift",
        "Distributed particles",
        "Write reconstructed multiplier and quadrature weights on the physical surface.",
        "ModalLift",
        "write"));
  }
} // namespace ImmersX::Coral
#endif
