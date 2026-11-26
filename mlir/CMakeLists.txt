cmake_minimum_required(VERSION 3.20)
project(AetherMLIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AETHER_BUILD_STANDALONE_VERIFIER "Build standalone C++17 Aether MLIR structural verifier (no LLVM required)" ON)

if(AETHER_BUILD_STANDALONE_VERIFIER)
  add_executable(aether-ir-verify tools/aether-ir-verify/aether_ir_verify.cpp)
  target_compile_options(aether-ir-verify PRIVATE -Wall -Wextra -O2)
endif()

# Out-of-Tree LLVM / MLIR Dialect Build (enabled automatically when MLIR_DIR is provided)
find_package(MLIR CONFIG QUIET)
if(MLIR_FOUND)
  message(STATUS "Found MLIRConfig.cmake in: ${MLIR_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

  add_subdirectory(include/Aether/Dialect)
  add_subdirectory(lib/Transforms)
  add_subdirectory(tools/aether-opt)
else()
  message(STATUS "MLIR_DIR not provided; built standalone aether-ir-verify. Pass -DMLIR_DIR=/path/to/lib/cmake/mlir to build aether-opt & TableGen dialects.")
endif()
