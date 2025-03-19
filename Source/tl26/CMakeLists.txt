# Define the library name
set(LIB_NAME tl26)

# Collect all source files in the current directory

# Include Subdirectories
include_directories(${PROJECT_SOURCE_DIR}/Source/API/)
include_directories(${PROJECT_SOURCE_DIR}/Source/Lib/Codec/)

set(SOURCES
tl26_flags.h
tl26_functions.c
tl26_feedback.h
tl26_feedback.c
tl26_request.h
tl26_request.c
tl26_python_thread.h  
tl26_python_thread.c  
)

set(CMAKE_C_VISIBILITY_PRESET default)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

add_library(${LIB_NAME} SHARED ${SOURCES})


find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Link Python libraries to the library
target_link_libraries(${LIB_NAME} PUBLIC ${Python3_LIBRARIES})

# Include Python headers for the library
target_include_directories(${LIB_NAME} PUBLIC ${Python3_INCLUDE_DIRS})

# add_library(${LIB_NAME} OBJECT ${SOURCES})
# Add the library
