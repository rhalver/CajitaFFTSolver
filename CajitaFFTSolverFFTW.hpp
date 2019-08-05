#ifndef CAJITA_FFT_FFTW_INCLUDED
#define CAJITA_FFT_FFTW_INCLUDED

#include "CajitaFFTSolver.hpp"

template< class ExecutionSpace, class MemorySpace> class CajitaFFTSolverFFTW : 
    public CajitaFFTSolver< ExecutionSpace, MemorySpace > 
{
    public:

        // once we know, what the required halo width for FFTW is, set the halo value
        // to the correct value
        
        /// constuctor to create a Cajita FFT Solver using the FFTW and
        /// initialize it with a grid from the application
        /// @param appGrid the grid that was created in the calling application
        CajitaFFTSolverFFTW(std::shared_ptr<Cajita::GlobalGrid> appGrid) :
            CajitaFFTSolver<ExecutionSpace, MemorySpace>(appGrid, fftw_halo_width)
        {
        }


        CajitaFFTSolverFFTW(
                            MPI_Comm comm,
                            std::vector<int> gridDim,
                            std::vector<bool> periodicity,
                            std::vector<double> frontCorner,
                            std::vector<double> backCorner
                        ) 
            : CajitaFFTSolver<ExecutionSpace, MemorySpace>
              (
                comm, 
                gridDim, 
                periodicity, 
                frontCorner, 
                backCorner, 
                fftw_halo_width
              )
        {
        }

        /// implementation for the FFTW class to bring the charges to the grid
        /// @param positions    particle positions
        /// @param charges      particle charges
        virtual void q2grid( Kokkos::View<double*, MemorySpace>, 
                             Kokkos::View<double*, MemorySpace> );

    private:
        static constexpr double fftw_halo_width = 0.0;
};

template< class ExecutionSpace, class MemorySpace > 
void CajitaFFTSolverFFTW<ExecutionSpace, MemorySpace>::q2grid
(
    Kokkos::View<double*, MemorySpace> positions,
    Kokkos::View<double*, MemorySpace> charges
)
{
    // bring charges to grid
    std::cout << "charges brought to grid" << std::endl;
}

#endif
