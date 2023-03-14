# CMake generated Testfile for 
# Source directory: /workspaces/CODICE_lagrangian/tests
# Build directory: /workspaces/CODICE_lagrangian/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests/template.debug "/usr/bin/cmake" "-DTRGT=tests.template.debug.test" "-DTEST=tests/template.debug" "-DEXPECT=PASSED" "-DBINARY_DIR=/workspaces/CODICE_lagrangian/build" "-DGUARD_FILE=/workspaces/CODICE_lagrangian/build/tests/template.debug/interrupt_guard.cc" "-P" "/usr/local/share/deal.II/scripts/run_test.cmake")
set_tests_properties(tests/template.debug PROPERTIES  LABEL "tests" TIMEOUT "600" WORKING_DIRECTORY "/workspaces/CODICE_lagrangian/build/tests/template.debug" _BACKTRACE_TRIPLES "/usr/local/share/deal.II/macros/macro_deal_ii_add_test.cmake;500;ADD_TEST;/usr/local/share/deal.II/macros/macro_deal_ii_pickup_tests.cmake;352;DEAL_II_ADD_TEST;/workspaces/CODICE_lagrangian/tests/CMakeLists.txt;2;DEAL_II_PICKUP_TESTS;/workspaces/CODICE_lagrangian/tests/CMakeLists.txt;0;")
add_test(tests/test_fourier_01.debug "/usr/bin/cmake" "-DTRGT=tests.test_fourier_01.debug.test" "-DTEST=tests/test_fourier_01.debug" "-DEXPECT=PASSED" "-DBINARY_DIR=/workspaces/CODICE_lagrangian/build" "-DGUARD_FILE=/workspaces/CODICE_lagrangian/build/tests/test_fourier_01.debug/interrupt_guard.cc" "-P" "/usr/local/share/deal.II/scripts/run_test.cmake")
set_tests_properties(tests/test_fourier_01.debug PROPERTIES  LABEL "tests" TIMEOUT "600" WORKING_DIRECTORY "/workspaces/CODICE_lagrangian/build/tests/test_fourier_01.debug" _BACKTRACE_TRIPLES "/usr/local/share/deal.II/macros/macro_deal_ii_add_test.cmake;500;ADD_TEST;/usr/local/share/deal.II/macros/macro_deal_ii_pickup_tests.cmake;352;DEAL_II_ADD_TEST;/workspaces/CODICE_lagrangian/tests/CMakeLists.txt;2;DEAL_II_PICKUP_TESTS;/workspaces/CODICE_lagrangian/tests/CMakeLists.txt;0;")
