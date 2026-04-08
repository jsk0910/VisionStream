cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(VisionStream LANGUAGES CXX CUDA)

# Setup CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Attempt to find pybind11, or use FetchContent if not found
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/csrc
)

# Core C++/CUDA library (csrc/)
add_library(visionstream_core SHARED
    csrc/memory/vision_buffer.cu
    csrc/graph/graph_manager.cpp
    csrc/codec/arithmetic_coder.cu
)
target_include_directories(visionstream_core PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

# Python bindings (visionstream/bindings/)
pybind11_add_module(visionstream_python MODULE
    visionstream/bindings/bindings.cpp
)

target_link_libraries(visionstream_python PRIVATE visionstream_core)

# Output as 'visionstream.so' for direct Python import
set_target_properties(visionstream_python PROPERTIES OUTPUT_NAME "visionstream")
