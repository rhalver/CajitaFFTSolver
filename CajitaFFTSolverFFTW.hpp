#ifndef CAJITA_FFT_FFTW_INCLUDED
#define CAJITA_FFT_FFTW_INCLUDED

#include "CajitaFFTSolver.hpp"

#include "fftw3.h"
#include "fftw3-mpi.h"

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

        virtual void forwardFFT() {};

        virtual void backwardFFT();

        virtual void grid2E( Kokkos::View<double*, MemorySpace> ) {}
        virtual void grid2F( Kokkos::View<double*, MemorySpace> ) {}
    private:
        static constexpr double fftw_halo_width = 0.0;
};


template< class ExecutionSpace, class MemorySpace >
void CajitaFFTSolverFFTW<ExecutionSpace, MemorySpace>::backwardFFT()
{
    // get private variables from the base class  
    auto grid = this->getGrid();
    auto layout = this->getLayout();
    auto array = this->getArray();
    auto gridDim = this->getGridDim();
    // create get view and cellsize
    auto view = array->view();
    auto cellSize = grid->cellSize();
    auto n_grid_points = gridDim(0) * gridDim(1) * gridDim(2);

    // allocate FFTW data type arrays
    fftw_complex* Qinput = new fftw_complex[n_grid_points];
    //fftw_complex* Qresult = new fftw_complex[n_grid_points];

    /*
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t alloc_local = fftw_mpi_local_size_3d(
            (int)view.extent(0),
            (int)view.extent(1),
            (int)view.extent(2),
            MPI_COMM_WORLD,
            &local_n0,
            &local_0_start
        );
    fftw_complex* Qinput = fftw_alloc_complex(alloc_local); 
    */

    // copy data to Qr array
    Kokkos::parallel_for
    (
        n_grid_points,
        KOKKOS_LAMBDA(int idx)
        {
            int cz = idx % gridDim(2);
            int cy = ( idx / gridDim(2) ) % gridDim(1);
            int cx = idx / ( gridDim(1) * gridDim(2) );

            Qinput[idx][0] = view(cx,cy,cz,0);
            Qinput[idx][1] = view(cx,cy,cz,0);
        }
    );


    // create FFTW plan
    fftw_plan plan;

    plan = fftw_mpi_plan_dft_3d((int)view.extent(0), 
                            (int)view.extent(1), 
                            (int)view.extent(2),
                            Qinput,
                            Qinput,
                            MPI_COMM_WORLD,
                            FFTW_BACKWARD,
                            FFTW_ESTIMATE);

   
    // execute FFTW plan
    fftw_execute(plan);
    
    
    // copy results to grid
    Kokkos::parallel_for
    (
        n_grid_points,
        KOKKOS_LAMBDA(int idx)
        {
            int cz = idx % gridDim(2);
            int cy = ( idx / gridDim(2) ) % gridDim(1);
            int cx = idx / ( gridDim(1) * gridDim(2) );

            view(cx,cy,cz,2) = Qinput[idx][0];
            view(cx,cy,cz,3) = Qinput[idx][1];

//            view(cx,cy,cz,2) = Qresult[idx][0];
//            view(cx,cy,cz,3) = Qresult[idx][1];
        }
    );

    //fftw_destroy_plan(plan);

    //delete [] Qresult;
    delete [] Qinput;
}
#endif
