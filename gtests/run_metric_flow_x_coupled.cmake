set(_command "${TEST_EXECUTABLE}" "${PARAMETER_FILE}")
if(DEFINED MPIEXEC AND NOT MPIEXEC STREQUAL "")
  list(PREPEND _command "${MPIEXEC}" -np 2)
endif()
execute_process(COMMAND ${_command}
                RESULT_VARIABLE _result
                OUTPUT_VARIABLE _stdout
                ERROR_VARIABLE _stderr
                TIMEOUT 590)
message("${_stdout}${_stderr}")
if(NOT "${_result}" STREQUAL "0")
  message(FATAL_ERROR "MetricFlowX coupled application failed: ${_result}")
endif()
if(NOT _stdout MATCHES "accepted IDA steps: [1-9][0-9]*")
  message(FATAL_ERROR "MetricFlowX coupled application did not advance in time")
endif()
