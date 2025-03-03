#
# Copyright(c) 2019 Intel Corporation
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
#

# APP Directory CMakeLists.txt

# Include API headers
include_directories(${PROJECT_SOURCE_DIR}/Source/API/)

set(all_files
    ../API/EbDebugMacros.h
    ../API/EbSvtAv1.h
    ../API/EbSvtAv1Enc.h
    ../API/EbSvtAv1ErrorCodes.h
    ../API/EbSvtAv1Formats.h
    ../API/EbSvtAv1Metadata.h
    app_config.c
    app_config.h
    app_context.c
    app_context.h
    app_input_y4m.c
    app_input_y4m.h
    app_main.c
    app_output_ivf.c
    app_output_ivf.h
    app_process_cmd.c
    svt_time.c
    svt_time.h
    tl26_functions.c
    tl26_flags.h
    )

# Python src
add_executable(SvtAv1EncApp ${all_files})
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

# Link the executable to shared library and Python libraries
target_link_libraries(SvtAv1EncApp
    SvtAv1Enc
    ${Python3_LIBRARIES}
)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    target_link_libraries(SvtAv1EncApp ${PLATFORM_LIBS})
elseif(UNIX)
    target_link_libraries(SvtAv1EncApp
        pthread
        m
    )
endif()

if(TARGET safestringlib)
    target_sources(SvtAv1EncApp PRIVATE $<TARGET_OBJECTS:safestringlib>)
endif()

# On macOS, set the shared library (SvtAv1Enc) to use dynamic lookup for Python symbols
if(APPLE)
    if(TARGET SvtAv1Enc)
        set_target_properties(SvtAv1Enc PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
endif()

install(TARGETS SvtAv1EncApp RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})
