cmake_minimum_required(VERSION 3.12)
project(cgal_simplify)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CGAL
find_package(CGAL REQUIRED COMPONENTS Core)

if(NOT CGAL_FOUND)
  message(FATAL_ERROR "CGAL not found. Please install CGAL.")
endif()

# Create executable
add_executable(cgal_simplify cgal_simplify.cpp)

# Link CGAL
target_link_libraries(cgal_simplify CGAL::CGAL)

# Set output directory for easier access
set_target_properties(cgal_simplify PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Platform-specific settings
if(MSVC)
    # Enable parallel compilation on Visual Studio
    target_compile_options(cgal_simplify PRIVATE /MP)
    # Set runtime library
    set_property(TARGET cgal_simplify PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Print build information
message(STATUS "")
message(STATUS "========================================")
message(STATUS "CGAL Mesh Simplification Tool")
message(STATUS "========================================")
message(STATUS "CGAL version: ${CGAL_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "========================================")
message(STATUS "")