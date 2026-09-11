cmake_minimum_required(VERSION 3.20)

project(Raytraced_audio LANGUAGES CXX C)
# ============================================================
# Language standards
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(EXTERNAL_DIR
    ${CMAKE_SOURCE_DIR}/external
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Dependencies
# ============================================================

# ------------------------------------------------------------
# box3d
# ------------------------------------------------------------

find_package(
    box3d
    0.1
    REQUIRED
    PATHS
        "C:/box3d"
)

# ------------------------------------------------------------
# Prebuilt MinGW raylib
#
# Headers:
#   C:/raylib/raylib/src
#
# Library:
#   C:/raylib/raylib/src/libraylib.a
#
# RUI::Rui has a PUBLIC dependency on the CMake target
# "raylib", so the target must exist before find_package(Rui).
# ------------------------------------------------------------

set(
    RAYLIB_DIR
    "C:/raylib/raylib"
)

set(
    RAYLIB_INCLUDE_DIR
    "${RAYLIB_DIR}/src"
)

set(
    RAYLIB_LIBRARY
    "${RAYLIB_DIR}/src/libraylib.a"
)

if(NOT EXISTS "${RAYLIB_INCLUDE_DIR}/raylib.h")
    message(FATAL_ERROR
        "raylib headers were not found at:\n"
        "  ${RAYLIB_INCLUDE_DIR}"
    )
endif()

if(NOT EXISTS "${RAYLIB_LIBRARY}")
    message(FATAL_ERROR
        "Prebuilt MinGW raylib library was not found at:\n"
        "  ${RAYLIB_LIBRARY}"
    )
endif()

# ------------------------------------------------------------
# Imported raylib target
# ------------------------------------------------------------

if(NOT TARGET raylib)

    add_library(
        raylib
        STATIC
        IMPORTED
        GLOBAL
    )

    set_target_properties(
        raylib
        PROPERTIES

            IMPORTED_LOCATION
                "${RAYLIB_LIBRARY}"

            INTERFACE_INCLUDE_DIRECTORIES
                "${RAYLIB_INCLUDE_DIR}"
    )

endif()

if(NOT TARGET raylib)
    message(FATAL_ERROR
        "Failed to create the prebuilt raylib target."
    )
endif()

# ------------------------------------------------------------
# RUI
# ------------------------------------------------------------

find_package(
    Rui
    CONFIG
    REQUIRED
    PATHS
        "C:/Libraries/rui"
)

# ============================================================
# Source discovery
# ============================================================

file(
    GLOB_RECURSE
    SRC_CPP
    CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

file(
    GLOB_RECURSE
    SRC_C
    CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.c
)

add_executable(
    ${PROJECT_NAME}
    ${SRC_CPP}
    ${SRC_C}
)

# ============================================================
# Include paths
# ============================================================

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(
    ${PROJECT_NAME}
    SYSTEM
    PRIVATE
        ${EXTERNAL_DIR}/imgui
        ${EXTERNAL_DIR}/nlohmann
)

# Add raylib's external directory to expose glad.h for raw OpenGL calls (UBO support)
set(
    RAYLIB_EXTERNAL_DIR
    "${RAYLIB_INCLUDE_DIR}/external"
)

if(NOT EXISTS "${RAYLIB_EXTERNAL_DIR}/glad.h")
    message(FATAL_ERROR
        "glad.h was not found at:\n"
        "  ${RAYLIB_EXTERNAL_DIR}/glad.h"
    )
endif()

target_include_directories(
    ${PROJECT_NAME}
    SYSTEM
    PRIVATE
        ${RAYLIB_EXTERNAL_DIR}
)

# ============================================================
# Compiler flags
# ============================================================

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE

        -w

        # ----------------------------------------------------
        # C++ Debug
        # ----------------------------------------------------

        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Og>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g1>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-fno-var-tracking>
        # ----------------------------------------------------
        # C++ Release
        # ----------------------------------------------------

        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-DNDEBUG>

        # ----------------------------------------------------
        # C Debug
        # ----------------------------------------------------

        $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-Og>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-g1>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-fno-var-tracking>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-fno-var-tracking-assignments>

        # ----------------------------------------------------
        # C Release
        # ----------------------------------------------------

        $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:-O2>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:-DNDEBUG>
)

# ============================================================
# Linking
# ============================================================
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        Rui::Rui
        box3d::box3d
        raylib

        gdi32
        ole32
        comdlg32
        opengl32
        shell32
        winmm
)
# ============================================================
# Windows / MinGW
# ============================================================

if(WIN32)

    target_link_options(
        ${PROJECT_NAME}
        PRIVATE

            $<$<CONFIG:Release>:-static-libgcc>
            $<$<CONFIG:Release>:-static-libstdc++>
            $<$<CONFIG:Release>:-mwindows>
    )

endif()

# ============================================================
# Distribution package
# ============================================================

set(
    RUNTIME_DIRS
    Ui
    shaders
    palletes
)

set(
    DIST_DIR
    ${CMAKE_BINARY_DIR}/dist/${PROJECT_NAME}
)

add_custom_target(
    dist

    COMMAND
        ${CMAKE_COMMAND}
        -E make_directory
        ${DIST_DIR}

    COMMAND
        ${CMAKE_COMMAND}
        -E copy
        $<TARGET_FILE:${PROJECT_NAME}>
        ${DIST_DIR}/

    COMMENT
        "Creating portable release package in ${DIST_DIR}"

    VERBATIM
)

add_dependencies(
    dist
    ${PROJECT_NAME}
)

foreach(dir ${RUNTIME_DIRS})

    if(EXISTS "${CMAKE_SOURCE_DIR}/${dir}")

        add_custom_command(
            TARGET
                dist

            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E copy_directory
                "${CMAKE_SOURCE_DIR}/${dir}"
                "${DIST_DIR}/${dir}"

            COMMENT
                "Copying ${dir}"
        )

    endif()

endforeach()

# ============================================================
# Run targets
# ============================================================

add_custom_target(
    run

    COMMAND
        $<TARGET_FILE:${PROJECT_NAME}>

    DEPENDS
        ${PROJECT_NAME}

    WORKING_DIRECTORY
        ${CMAKE_SOURCE_DIR}

    USES_TERMINAL
)

add_custom_target(
    run-dist

    COMMAND
        "${DIST_DIR}/${PROJECT_NAME}.exe"

    DEPENDS
        dist

    WORKING_DIRECTORY
        ${DIST_DIR}

    USES_TERMINAL
)

# ============================================================
# Configuration summary
# ============================================================

message(STATUS "")
message(STATUS "================================================")
message(STATUS "Procedural Game")
message(STATUS "================================================")
message(STATUS "Compiler:           ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C Compiler:         ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler:       ${CMAKE_CXX_COMPILER}")
message(STATUS "Raylib include:     ${RAYLIB_INCLUDE_DIR}")
message(STATUS "Raylib library:     ${RAYLIB_LIBRARY}")
message(STATUS "RUI:                C:/Libraries/rui")
message(STATUS "box3d:              C:/box3d")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "================================================")
message(STATUS "")