# Header‑only but still adds a target

if(${FROM_PIP_INSTALL}) 
    message(STATUS "Building for pip install")
    cmake_minimum_required(VERSION 3.20)
    project(av1_gym_bridge LANGUAGES C CXX ASM)

    # Global compile options
    set(CMAKE_C_STANDARD   11)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_C_FLAGS "-O0 ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 ${CMAKE_CXX_FLAGS}")

    add_compile_definitions(SVT_ENABLE_USER_CALLBACKS)

    # Compile in the upstream SVT‑AV1 LIB
    set(BUILD_SHARED_LIBS OFF  CACHE BOOL "" FORCE)
    set(BUILD_APPS        OFF  CACHE BOOL "" FORCE)

    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/svtcore
        EXCLUDE_FROM_ALL)
endif()

find_package(Threads  REQUIRED)
find_package(Python3  REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED) 

# Add Safe‑string library
file(GLOB SAFE_SRC "${CMAKE_CURRENT_LIST_DIR}/../third_party/safestringlib/*.c")
add_library(safeclib STATIC ${SAFE_SRC})
target_include_directories(safeclib PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/safestringlib")

# Compole plugin callbacks
add_library(svtav1_plugin STATIC
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib/Globals/enc_callbacks.c")

target_include_directories(svtav1_plugin PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/.."
    "${CMAKE_CURRENT_LIST_DIR}/../Source"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/API")

target_link_libraries(svtav1_plugin PRIVATE Threads::Threads)

# Re‑build Source/App/ as a static library, renaming main()
file(GLOB APP_SRC "${CMAKE_CURRENT_LIST_DIR}/../Source/App/*.c")
add_library(svtav1_app STATIC ${APP_SRC})

target_compile_definitions(svtav1_app PRIVATE main=svt_enc_app_main)
target_include_directories(svtav1_app PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/.."
    "${CMAKE_CURRENT_LIST_DIR}/../Source"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/App"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/API")

target_link_libraries(svtav1_app PUBLIC Threads::Threads safeclib)

# Build the internal Python extension _av1_wrapper.{so,dylib,pyd}
set(BRIDGE_SRC
    src/av1gym/pyencoder/bridge/pybridge.cpp
    src/av1gym/pyencoder/bridge/cb_registration.cpp
    src/av1gym/pyencoder/bridge/utils.cpp)

set(PYENCODER_SRC
    src/av1gym/pyencoder/_binding.cpp
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib/Globals/rl_feedback.c")

add_library(_av1_wrapper MODULE
    ${BRIDGE_SRC}
    ${PYENCODER_SRC})

# On Windows a Python extension must be named *.pyd without the "lib" prefix.
if (WIN32)
    set_target_properties(_av1_wrapper PROPERTIES
        OUTPUT_NAME "_av1_wrapper"
        PREFIX      ""
        SUFFIX      ".pyd")
else()
    set_target_properties(_av1_wrapper PROPERTIES PREFIX "")
endif()

# Include paths
target_include_directories(_av1_wrapper PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/.."
    src/av1gym/pyencoder/bridge
    "${CMAKE_CURRENT_LIST_DIR}/../Source"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/App"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/API"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib/Globals"
    ${Python3_INCLUDE_DIRS})

# Link targets
target_link_libraries(_av1_wrapper
    PRIVATE
        pybind11::module
        svtav1_plugin
        svtav1_app
        SvtAv1Enc
        safeclib
        Python3::Python
        Threads::Threads
        m)                   # libm on *nix

# Installation path for scikit‑build‑core
if(${FROM_PIP_INSTALL})
    install(TARGETS _av1_wrapper
        LIBRARY DESTINATION av1gym/pyencoder
        RUNTIME DESTINATION av1gym/pyencoder)
else()
    install(TARGETS _av1_wrapper
        LIBRARY DESTINATION ${Python3_SITEARCH}/av1gym/pyencoder
        RUNTIME DESTINATION ${Python3_SITEARCH}/av1gym/pyencoder)
endif()
