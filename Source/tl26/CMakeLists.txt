# Define the library name
set(LIB_NAME tl26)

# Collect all source files in the current directory

# Include Subdirectories
include_directories(${PROJECT_SOURCE_DIR}/Source/API/)

set(SOURCES
tl26_feedback.c
tl26_feedback.h
tl26_flags.h
tl26_functions.c
tl26_request.c
tl26_request.h
)


find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# Add the library
add_library(${LIB_NAME} OBJECT ${SOURCES})

# Link Python libraries to the library
target_link_libraries(${LIB_NAME} PRIVATE ${Python3_LIBRARIES})

# Include Python headers for the library
target_include_directories(${LIB_NAME} PRIVATE ${Python3_INCLUDE_DIRS})

# Specify include directories for this library
# target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} /home/tom/tl26/SVT-AV1/Source/Lib/APP)
