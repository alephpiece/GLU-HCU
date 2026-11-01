# The tiny diagonal system has the exactly representable solution x=1.
file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
file(WRITE "${TEST_DIRECTORY}/identity.mtx" "2 2 2\n1 1 1\n2 2 1\n")

function(run_cli name expected_status expected_diagnostic)
    execute_process(COMMAND "${GLU_EXECUTABLE}" -i "${TEST_DIRECTORY}/identity.mtx"
        WORKING_DIRECTORY "${TEST_DIRECTORY}/${name}"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error
        TIMEOUT 30)
    file(WRITE "${TEST_DIRECTORY}/${name}.log" "${output}${error}")
    if(NOT "${status}" STREQUAL "${expected_status}")
        message(FATAL_ERROR "${name}: expected exit ${expected_status}, got ${status}: ${error}")
    endif()
    if(NOT "${expected_diagnostic}" STREQUAL "" AND NOT error MATCHES "${expected_diagnostic}")
        message(FATAL_ERROR "${name}: missing diagnostic: ${error}")
    endif()
endfunction()

file(MAKE_DIRECTORY "${TEST_DIRECTORY}/success")
file(REMOVE "${TEST_DIRECTORY}/success/x.dat")
run_cli(success 0 "")
file(READ "${TEST_DIRECTORY}/success/x.dat" solution)
if(NOT solution STREQUAL "1\n1\n")
    message(FATAL_ERROR "Unexpected identity solution: ${solution}")
endif()

# A directory named x.dat reliably rejects opening, even when running as root.
file(MAKE_DIRECTORY "${TEST_DIRECTORY}/open-failure/x.dat")
run_cli(open-failure 1 "Failed to open x.dat")

# /dev/full permits opening but rejects writes, including buffered close.
file(MAKE_DIRECTORY "${TEST_DIRECTORY}/write-failure")
file(REMOVE "${TEST_DIRECTORY}/write-failure/x.dat")
file(CREATE_LINK /dev/full "${TEST_DIRECTORY}/write-failure/x.dat" SYMBOLIC)
run_cli(write-failure 1 "Failed to write x.dat")

file(MAKE_DIRECTORY "${TEST_DIRECTORY}/no-device")
file(REMOVE "${TEST_DIRECTORY}/no-device/x.dat")
set(ENV{CUDA_VISIBLE_DEVICES} "-1")
set(ENV{HIP_VISIBLE_DEVICES} "-1")
run_cli(no-device 1 "")
if(EXISTS "${TEST_DIRECTORY}/no-device/x.dat")
    message(FATAL_ERROR "GPU failure produced a solution file")
endif()
