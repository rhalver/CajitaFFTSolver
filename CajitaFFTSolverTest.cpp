#include "CajitaFFTSolverFFTW.hpp"

#include <random>
#include "fftw3.h"
#include "fftw3-mpi.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    fftw_mpi_init();
    Kokkos::initialize(argc,argv);
    {
        const int n_part_total = 300;
        const double low_corner = 0.0;
        const double high_corner = 100.0;
        const int n_cells = 3;

        int n_ranks;
        MPI_Comm_size(MPI_COMM_WORLD,&n_ranks);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);

        CajitaFFTSolver<Kokkos::OpenMP, Kokkos::HostSpace>* cffts =
            new CajitaFFTSolverFFTW<Kokkos::OpenMP,Kokkos::HostSpace>
            (
                MPI_COMM_WORLD,
                std::vector<int>({n_cells,n_cells,n_cells}),
                std::vector<bool>({true,true,true}),
                std::vector<double>({low_corner,low_corner,low_corner}),
                std::vector<double>({high_corner,high_corner,high_corner})
            );
        auto index = (cffts->getLocalLayout())->indexSpace(Cajita::Own(), Cajita::Global());
        auto cellSize = (high_corner - low_corner) / (double)n_cells;

        double factor = (index.max(0) - index.min(0)) *
                        (index.max(1) - index.min(1)) *
                        (index.max(2) - index.min(2)) /
                        (double)(n_cells * n_cells * n_cells);
        int n_part = (int)( (double)n_part_total * factor);
        
        Kokkos::View<double*, Kokkos::OpenMP> r("positions", 3 * n_part);
        Kokkos::View<double*, Kokkos::OpenMP> q("positions", n_part);

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution_rx(index.min(0)*cellSize,index.max(0)*cellSize);
        std::uniform_real_distribution<double> distribution_ry(index.min(1)*cellSize,index.max(1)*cellSize);
        std::uniform_real_distribution<double> distribution_rz(index.min(2)*cellSize,index.max(2)*cellSize);
        std::uniform_real_distribution<double> distribution_q(-1,1);

        // distribute particles randomly in the system
        for (int i = 0; i < n_part; ++i)
        {
            r(3*i+0) = distribution_rx(generator);
            r(3*i+1) = distribution_ry(generator);
            r(3*i+2) = distribution_rz(generator);
            q(i) = distribution_q(generator);
        }


        // normalize charge (periodic system should be neutral!)
        double total_charge = 0.0;
        Kokkos::parallel_reduce( n_part, KOKKOS_LAMBDA ( const int i, double& part_charge )
                {
                    part_charge += q(i);
                }, total_charge);
        total_charge /= (double)n_part;
        Kokkos::parallel_for( n_part, KOKKOS_LAMBDA (const int i)
                {
                    q(i) -= total_charge;
                });

        // bring charges to grid
        cffts->q2grid(r, q);
        cffts->backwardFFT();
    }
    Kokkos::finalize();
    fftw_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
