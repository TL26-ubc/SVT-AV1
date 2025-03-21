find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Corrected variable usage
message(STATUS "new_package: ${Python3_LIBRARIES}")
message(STATUS "new_package: ${Python3_INCLUDE_DIRS}")

# Function to link Python libraries
function(link_python_library target)
    target_link_libraries(${target} PRIVATE ${Python3_LIBRARIES})
    target_include_directories(${target} PRIVATE ${Python3_INCLUDE_DIRS})
endfunction()

# Function to include Python headers
function(include_python_headers target)
    target_include_directories(${target} PRIVATE ${Python3_INCLUDE_DIRS})
endfunction()
