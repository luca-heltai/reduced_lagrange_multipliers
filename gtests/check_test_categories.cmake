# Validate the compiled registry, including instantiated parameterized suites,
# rather than comparing two hand-maintained inventories.
include("${CATEGORY_FILE}")
execute_process(
  COMMAND "${TEST_EXECUTABLE}" --gtest_list_tests --gtest_filter=*
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _registry
  ERROR_VARIABLE _stderr
  TIMEOUT 30)
if(NOT _result STREQUAL "0")
  message(FATAL_ERROR "GoogleTest discovery failed: ${_result}\n${_stderr}")
endif()

string(REPLACE "\r" "" _registry "${_registry}")
string(REPLACE "\n" ";" _lines "${_registry}")
set(_suite_count 0)
foreach(_line IN LISTS _lines)
  if(_line MATCHES "^([^ .]+)\\.[ ]*(#.*)?$")
    set(_suite "${CMAKE_MATCH_1}")
    set(_matches 0)
    foreach(_category IN LISTS IMMERSX_TEST_PRIMARY_CATEGORIES)
      string(TOUPPER "${_category}" _upper)
      foreach(_pattern IN LISTS IMMERSX_${_upper}_TEST_SUITES)
        # Use the same glob syntax as GoogleTest, not raw regular expressions.
        string(REGEX REPLACE "([][+.^$(){}|\\\\])" "\\\\\\1" _regex "${_pattern}")
        string(REPLACE "*" ".*" _regex "${_regex}")
        string(REPLACE "?" "." _regex "${_regex}")
        if(_suite MATCHES "^${_regex}$")
          math(EXPR _matches "${_matches} + 1")
        endif()
      endforeach()
    endforeach()
    if(NOT _matches EQUAL 1)
      message(FATAL_ERROR
        "Compiled GoogleTest suite '${_suite}' matches ${_matches} categories; expected exactly one")
    endif()
    math(EXPR _suite_count "${_suite_count} + 1")
  endif()
endforeach()
if(_suite_count EQUAL 0)
  message(FATAL_ERROR "No compiled GoogleTest suites were discovered")
endif()
message(STATUS "Classified all ${_suite_count} compiled GoogleTest suites")
