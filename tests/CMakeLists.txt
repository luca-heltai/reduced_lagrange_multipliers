INCLUDE_DIRECTORIES(${D2K_INCLUDE_DIRS})

add_compile_options(
  $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>)

DEAL_II_PICKUP_TESTS()

# Deal.II regression tests use the same primary categories and quick policy as
# GoogleTests.  These fixtures exercise reusable numerical components;
# the MPI matrix-free transfer test is additionally marked with mpi.
get_property(_dealii_tests DIRECTORY PROPERTY TESTS)
foreach(_dealii_test ${_dealii_tests})
  set(_dealii_labels "integration;quick")
  if(_dealii_test MATCHES "matrix_free_transfers\.mpirun")
    list(APPEND _dealii_labels mpi)
  elseif(_dealii_test MATCHES "test_fourier")
    set(_dealii_labels "unit;quick")
  endif()
  # Regression cases build their executables through the shared build tree.
  set_tests_properties(${_dealii_test} PROPERTIES LABELS "${_dealii_labels}"
                                                 RUN_SERIAL TRUE)
endforeach()
