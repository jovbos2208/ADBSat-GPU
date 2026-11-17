cmake_minimum_required(VERSION 3.22)
project(ADBSatGPU VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ADBSAT_FETCH_KOKKOS "Download pinned Kokkos when no package is installed" ON)
option(ADBSAT_BUILD_TESTS "Build correctness tests" ON)
option(ADBSAT_BUILD_BENCHMARKS "Build benchmark executable" ON)

find_package(Kokkos 5.2 QUIET)
if(NOT Kokkos_FOUND)
  if(NOT ADBSAT_FETCH_KOKKOS)
    message(FATAL_ERROR "Kokkos >= 5.2 not found and ADBSAT_FETCH_KOKKOS=OFF")
  endif()
  include(FetchContent)
  FetchContent_Declare(kokkos
    URL https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2.tar.gz
    URL_HASH SHA256=d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(kokkos)
endif()

add_library(adbsat_gpu src/core.cpp)
target_include_directories(adbsat_gpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(adbsat_gpu PUBLIC Kokkos::kokkos)
target_compile_options(adbsat_gpu PRIVATE $<$<CONFIG:Release>:-O3>)

add_executable(adbsat-cli src/cli.cpp)
target_link_libraries(adbsat-cli PRIVATE adbsat_gpu)

if(ADBSAT_BUILD_TESTS)
  enable_testing()
  add_executable(adbsat-tests tests/test_core.cpp)
  target_link_libraries(adbsat-tests PRIVATE adbsat_gpu)
  add_test(NAME adbsat-correctness COMMAND adbsat-tests)
endif()

if(ADBSAT_BUILD_BENCHMARKS)
  add_executable(adbsat-benchmark benchmarks/benchmark.cpp)
  target_link_libraries(adbsat-benchmark PRIVATE adbsat_gpu)
  add_executable(adbsat-shadow-benchmark benchmarks/shadow_benchmark.cpp)
  target_link_libraries(adbsat-shadow-benchmark PRIVATE adbsat_gpu)
endif()

install(TARGETS adbsat_gpu adbsat-cli)
install(DIRECTORY include/ DESTINATION include)
