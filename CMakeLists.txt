cmake_minimum_required(VERSION 3.12)

project(CajitaFFTSolverTesting LANGUAGES CXX VERSION 0.0.1)

#find_package(PkgConfig)
find_package(MPI REQUIRED)
#pkg_check_modules(KOKKOS kokkos IMPORTED_TARGET)
find_package(Kokkos REQUIRED)
find_package(Cajita REQUIRED)
find_package(FFTW3 REQUIRED)

add_executable (CajitaFFTSolverTest CajitaFFTSolver.hpp CajitaFFTSolverFFTW.hpp CajitaFFTSolverTest.cpp)

target_link_libraries(CajitaFFTSolverTest Cajita::Cajita Kokkos::Kokkos MPI::MPI_CXX)
#target_include_directories( ArrayTest 
#                            $<INSTALL_INTERFACE:include>
#                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
#                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#                          )
