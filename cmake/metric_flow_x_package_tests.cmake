# These configure-only tests inspect the actual ImmersX library link commands.
# They use isolated fixture packages, so CI does not need MetricFlowX installed.
function(_immersx_metric_package_test package build_type enabled)
  set(_name "ImmersX.integration.MetricFlowXPackage.${package}.${build_type}")
  set(_deal_libraries)
  foreach(_variant Debug Release)
    string(TOUPPER "${_variant}" _upper)
    if(_upper IN_LIST DEAL_II_BUILD_TYPES)
      list(APPEND _deal_libraries
        "-DDEAL_LIBRARY_${_variant}=$<TARGET_FILE:${DEAL_II_TARGET_${_upper}}>")
    endif()
  endforeach()
  add_test(NAME "${_name}"
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DBINARY_DIR=${CMAKE_BINARY_DIR}/metric_package_tests/${package}-${build_type}"
      "-DDEAL_II_DIR=${deal.II_DIR}"
      "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DPACKAGE=${package}" "-DBUILD_TYPE=${build_type}"
      "-DEXPECT_ENABLED=${enabled}"
      ${_deal_libraries}
      -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_metric_flow_x_package_test.cmake")
  set_tests_properties("${_name}" PROPERTIES
    LABELS "integration;quick" TIMEOUT 180)
endfunction()

foreach(_variant Debug Release)
  string(TOUPPER "${_variant}" _upper)
  if(_upper IN_LIST DEAL_II_BUILD_TYPES)
    _immersx_metric_package_test(Dual ${_variant} ON)
    _immersx_metric_package_test(Legacy${_variant} ${_variant} ON)
    _immersx_metric_package_test(Unconfigured${_variant} ${_variant} ON)
    _immersx_metric_package_test(LegacyDual ${_variant} ON)
  endif()
endforeach()
if("DEBUG" IN_LIST DEAL_II_BUILD_TYPES AND
   "RELEASE" IN_LIST DEAL_II_BUILD_TYPES)
  _immersx_metric_package_test(LegacyDebug Release OFF)
  _immersx_metric_package_test(LegacyRelease Debug OFF)
  _immersx_metric_package_test(Dual DebugRelease ON)
  _immersx_metric_package_test(ConcreteOnly DebugRelease ON)
  foreach(_package LegacyDebug LegacyRelease LegacyDual Missing Partial)
    _immersx_metric_package_test(${_package} DebugRelease OFF)
  endforeach()
endif()
