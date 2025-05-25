cmake_minimum_required(VERSION 3.20)
project(av1_gym_bridge LANGUAGES C CXX ASM)

add_compile_definitions(SVT_ENABLE_USER_CALLBACKS)

# 1.  Bring in the upstream SVT‑AV1 build (core only, static)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_APPS        OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/svtcore
    EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 2.  Safe‑string library  (strcpy_s, strncpy_s, …)
file(GLOB SAFE_SRC
     "${CMAKE_CURRENT_LIST_DIR}/../third_party/safestringlib/*.c")

add_library(safeclib STATIC ${SAFE_SRC})
target_include_directories(safeclib PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/safestringlib")

# 3.  Plugin callbacks (defines svt_av1_enc_set_callbacks / plugin_cbs)
add_library(svtav1_plugin STATIC
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib/Globals/enc_callbacks.c")

target_include_directories(svtav1_plugin PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/.."
    "${CMAKE_CURRENT_LIST_DIR}/../Source"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/API")

target_link_libraries(svtav1_plugin PRIVATE Threads::Threads)

# 4.  Re‑build Source/App/ as a static lib, renaming main()
file(GLOB APP_SRC "${CMAKE_CURRENT_LIST_DIR}/../Source/App/*.c")
add_library(svtav1_app STATIC ${APP_SRC})

target_compile_definitions(svtav1_app PRIVATE main=svt_enc_app_main)
target_include_directories(svtav1_app PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/.."
    "${CMAKE_CURRENT_LIST_DIR}/../Source"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/App"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/API")

target_link_libraries(svtav1_app PUBLIC Threads::Threads safeclib)

# 5.  Build the Python extension  _svtapp.{so,dylib,pyd}
add_library(_svtapp MODULE
    src/bridge/pybridge.c
    src/bridge/cb_registration.c
    src/bridge/cb_validation.c
    src/bridge/py_trampoline.c
    src/pyencoder/_binding.c
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib/Globals/rl_feedback.c")

if (WIN32)
    set_target_properties(_svtapp PROPERTIES
        OUTPUT_NAME "_svtapp"
        PREFIX ""
        SUFFIX ".pyd")
else()
    set_target_properties(_svtapp PROPERTIES PREFIX "")
endif()

target_include_directories(_svtapp PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/.."
    src/bridge
    "${CMAKE_CURRENT_LIST_DIR}/../Source"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/App"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/API"
    "${CMAKE_CURRENT_LIST_DIR}/../Source/Lib/Globals" 
    ${Python3_INCLUDE_DIRS})

target_link_libraries(_svtapp
    PRIVATE
        svtav1_plugin
        svtav1_app
        SvtAv1Enc
        safeclib
        Python3::Python
        Threads::Threads
        m)
# Install path for scikit‑build‑core editable wheels
install(TARGETS _svtapp LIBRARY DESTINATION . RUNTIME DESTINATION .)
