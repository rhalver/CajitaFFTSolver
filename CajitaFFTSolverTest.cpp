#include "CajitaFFTSolver.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc,argv);
    {
        CajitaFFTSolver<Kokkos::OpenMP,Kokkos::HostSpace> cffts
            (
                MPI_COMM_WORLD,
                std::vector<int>({25,25,25}),
                std::vector<bool>({true,true,true}),
                std::vector<double>({0.0,0.0,0.0}),
                std::vector<double>({100,100,100})
            );
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
