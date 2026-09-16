# simple-crast: an AVX2 software renderer for Crytek Sponza.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release     (single-config: Linux, Ninja, Makefiles)
#   cmake --build build --config Release
#
# Builds with MSVC on Windows and GCC or Clang on Linux. The only dependency is
# SDL3, compiled from SDL3/ as a static library (window, input, threads); JPEG
# and PNG decoding uses the stb_image.h that ships inside SDL's source tree.
# The executable, crast (crast.exe on Windows), lands in this directory; it
# finds the sponza/ folder next to itself or in a parent directory.
#
# Linux needs the usual SDL3 video prerequisites, e.g. on Debian/Ubuntu:
#   libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxss-dev
#   libwayland-dev libxkbcommon-dev wayland-protocols

cmake_minimum_required(VERSION 3.20)
project(simple-crast LANGUAGES C)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|amd64|x64)$")
  message(FATAL_ERROR "simple-crast needs an x86-64 target (it is written with AVX2 intrinsics).")
endif()
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")   # one self-contained exe
endif()

# ---------------------------------------------------------------------------
# SDL3: static, with the video subsystem and nothing else the renderer does
# not use. The render API stays on: some video backends (Wayland) implement
# the window surface on top of it.
# ---------------------------------------------------------------------------
function(simple_crast_force var value)
  set(${var} ${value} CACHE BOOL "Forced by simple-crast" FORCE)
endfunction()
simple_crast_force(SDL_SHARED       OFF)
simple_crast_force(SDL_STATIC       ON)
simple_crast_force(SDL_TEST_LIBRARY OFF)
simple_crast_force(SDL_TESTS        OFF)
simple_crast_force(SDL_EXAMPLES     OFF)
simple_crast_force(SDL_INSTALL      OFF)
simple_crast_force(SDL_UNINSTALL    OFF)
simple_crast_force(SDL_AUDIO        OFF)
simple_crast_force(SDL_GPU          OFF)
simple_crast_force(SDL_CAMERA       OFF)
simple_crast_force(SDL_JOYSTICK     OFF)
simple_crast_force(SDL_HAPTIC       OFF)
simple_crast_force(SDL_HIDAPI       OFF)
simple_crast_force(SDL_POWER        OFF)
simple_crast_force(SDL_SENSOR       OFF)
simple_crast_force(SDL_DIALOG       OFF)
simple_crast_force(SDL_TRAY         OFF)
simple_crast_force(SDL_VULKAN       OFF)
add_subdirectory(SDL3 EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------
add_executable(simple-crast WIN32 src/renderer.c src/cpucheck.c src/image.c)
target_link_libraries(simple-crast PRIVATE SDL3::SDL3-static)
target_include_directories(simple-crast PRIVATE SDL3/src/video)   # stb_image.h
set_target_properties(simple-crast PROPERTIES
  OUTPUT_NAME crast                                        # crast.exe on Windows, crast on Linux
  C_STANDARD 11
  RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_CURRENT_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  target_compile_definitions(simple-crast PRIVATE _CRT_SECURE_NO_WARNINGS)
  # The unity TU: AVX2 + FMA, whole-program optimization.
  set_source_files_properties(src/renderer.c PROPERTIES
    COMPILE_OPTIONS "$<$<CONFIG:Release>:/O2;/Oi;/Ot;/GL;/fp:fast>;/arch:AVX2")
  set_source_files_properties(src/cpucheck.c src/image.c PROPERTIES
    COMPILE_OPTIONS "$<$<CONFIG:Release>:/O2>")
  target_link_options(simple-crast PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
  # GCC / Clang. -ffast-math is /fp:fast's counterpart; only the unity TU gets
  # AVX2 code generation, so cpucheck.c can still run on an older CPU.
  set_source_files_properties(src/renderer.c PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma;-ffast-math;$<$<CONFIG:Release>:-O2>")
  set_source_files_properties(src/cpucheck.c src/image.c PROPERTIES
    COMPILE_OPTIONS "$<$<CONFIG:Release>:-O2>")
  find_package(Threads REQUIRED)
  target_link_libraries(simple-crast PRIVATE Threads::Threads m)
endif()
