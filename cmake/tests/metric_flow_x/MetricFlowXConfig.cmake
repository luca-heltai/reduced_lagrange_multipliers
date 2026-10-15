# Model the public target shapes without compiling MetricFlowX. The files
# exist only to let CMake generate real library link commands in the consumer.
function(_metric_fixture_target target variant config)
  string(TOUPPER "${variant}" _upper)
  set(_library
    "${CMAKE_CURRENT_LIST_DIR}/libmetric_fixture_${variant}${CMAKE_SHARED_LIBRARY_SUFFIX}")
  file(WRITE "${_library}" "")
  add_library(${target} SHARED IMPORTED)
  if(config STREQUAL "NONE")
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${_library}")
  else()
    set_target_properties(${target} PROPERTIES
      IMPORTED_CONFIGURATIONS "${config}"
      IMPORTED_LOCATION_${config} "${_library}")
  endif()
  set_target_properties(${target} PROPERTIES
    INTERFACE_LINK_LIBRARIES "${DEAL_II_TARGET_${_upper}}")
endfunction()

if(METRIC_FIXTURE_PACKAGE STREQUAL "Dual" OR
   METRIC_FIXTURE_PACKAGE STREQUAL "ConcreteOnly")
  foreach(_variant Debug Release)
    _metric_fixture_target(MetricFlowX::metric_flow_x_${_variant}
                           ${_variant} DEBUGRELEASE)
  endforeach()
elseif(METRIC_FIXTURE_PACKAGE STREQUAL "Partial")
  _metric_fixture_target(MetricFlowX::metric_flow_x_Debug Debug DEBUGRELEASE)
elseif(METRIC_FIXTURE_PACKAGE STREQUAL "LegacyDebug")
  _metric_fixture_target(MetricFlowX::metric_flow_x Debug DEBUG)
elseif(METRIC_FIXTURE_PACKAGE STREQUAL "LegacyRelease")
  _metric_fixture_target(MetricFlowX::metric_flow_x Release RELEASE)
elseif(METRIC_FIXTURE_PACKAGE STREQUAL "UnconfiguredDebug")
  _metric_fixture_target(MetricFlowX::metric_flow_x Debug NONE)
elseif(METRIC_FIXTURE_PACKAGE STREQUAL "UnconfiguredRelease")
  _metric_fixture_target(MetricFlowX::metric_flow_x Release NONE)
endif()

if(METRIC_FIXTURE_PACKAGE STREQUAL "Dual" OR
   METRIC_FIXTURE_PACKAGE STREQUAL "LegacyDual" OR
   METRIC_FIXTURE_PACKAGE STREQUAL "Partial")
  # Like the real package, the generic selector uses the global configuration:
  # DebugRelease falls through to Release, including its deal.II dependency.
  add_library(MetricFlowX::metric_flow_x SHARED IMPORTED)
  foreach(_variant Release Debug)
    string(TOUPPER "${_variant}" _upper)
    set(_library
      "${CMAKE_CURRENT_LIST_DIR}/libmetric_fixture_${_variant}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    file(WRITE "${_library}" "")
    set_property(TARGET MetricFlowX::metric_flow_x APPEND PROPERTY
      IMPORTED_CONFIGURATIONS "${_upper}")
    set_property(TARGET MetricFlowX::metric_flow_x PROPERTY
      IMPORTED_LOCATION_${_upper} "${_library}")
    if(_variant STREQUAL "Debug")
      set(_condition "$<CONFIG:Debug>")
    elseif(_variant STREQUAL "Release")
      set(_condition "$<NOT:$<CONFIG:Debug>>")
    endif()
    set_property(TARGET MetricFlowX::metric_flow_x APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "$<${_condition}:${DEAL_II_TARGET_${_upper}}>")
  endforeach()
endif()
