#include "CajitaFFTSolverFFTW.hpp"

#include <random>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc,argv);
    {
        const int n_part = 30;

        CajitaFFTSolver<Kokkos::OpenMP, Kokkos::HostSpace>* cffts =
            new CajitaFFTSolverFFTW<Kokkos::OpenMP,Kokkos::HostSpace>
            (
                MPI_COMM_WORLD,
                std::vector<int>({25,25,25}),
                std::vector<bool>({true,true,true}),
                std::vector<double>({0.0,0.0,0.0}),
                std::vector<double>({100,100,100})
            );
        Kokkos::View<double*, Kokkos::OpenMP> r("positions", 3 * n_part);
        Kokkos::View<double*, Kokkos::OpenMP> q("positions", n_part);

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution_r(0,100);
        std::uniform_real_distribution<double> distribution_q(-1,1);

        // distribute particles randomly in the system
        for (int i = 0; i < n_part; ++i)
        {
            for (int d = 0; d < 3; ++d)
                r(3*i+d) = distribution_r(generator);
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
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
