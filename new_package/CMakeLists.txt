

# Function to link Python libraries
message(STATUS "Python3_LIBRARIES: ${PYTHON_LIBRARIES}")
function(link_python_library target)
    target_link_libraries(${target} PRIVATE ${PYTHON_LIBRARIES})
    target_include_directories(${target} PRIVATE ${PYTHON_INCLUDE_DIRS})
endfunction()

# Function to include Python headers
message(STATUS "Python3_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
function(include_python_headers target)
    target_include_directories(${target} PRIVATE ${PYTHON_INCLUDE_DIRS})
endfunction()
