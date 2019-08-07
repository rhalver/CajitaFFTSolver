#ifndef CAJITA_FFT_INCLUDED
#define CAJITA_FFT_INCLUDED

#include <mpi.h>
#include <assert.h>
#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_UniformDimPartitioner.hpp>
#include <Cajita_Array.hpp>
#include <Cajita_Halo.hpp>
#include <Cajita_BovWriter.hpp>

#include <Kokkos_Core.hpp>

#include <vector>
#include <iostream>

template <class ExecutionSpace, class MemorySpace> class CajitaFFTSolver 
{
    public:
        /// default constructor
        CajitaFFTSolver() 
        {
            gridDims = Kokkos::View<int*, MemorySpace>("gridDims", 3);
        };
        /// constructor to set the grid dimensions
        /// @param comm         MPI Communicator used by the application
        /// @param gridDim      number of grid points in each cartesian direction
        /// @param periodicity  periodicitiy in each cartesian direction
        /// @param frontCorner  the lower, left, front corner of the system
        /// @param backCorner   the upper, right, back corner of the system
        /// @param halo         the width of the halo region
        CajitaFFTSolver(
                            MPI_Comm comm,
                            std::vector<int> gridDim,
                            std::vector<bool> periodicity,
                            std::vector<double> frontCorner,
                            std::vector<double> backCorner,
                            int halo
                        ) 
            : CajitaFFTSolver()
        {
            // set up the partitioner (for now let MPI decide, later needs adjustments
            // for the FFT library specifics -> inheriting classes?)
            Cajita::UniformDimPartitioner partitioner;
            // store grid dimension to internal variable
            for (auto i = 0; i < 3; ++i)
            {
                gridDims(i) = gridDim.at(i);
            }
            // uniform cell size required (assert can be reqplace with nicer error checking later)
            assert
                (
                    ((backCorner.at(0) - frontCorner.at(0)) / (double)gridDims(0)) ==
                    ((backCorner.at(1) - frontCorner.at(1)) / (double)gridDims(1))
                );
            assert
                (
                    ((backCorner.at(0) - frontCorner.at(0)) / (double)gridDims(0)) ==
                    ((backCorner.at(2) - frontCorner.at(2)) / (double)gridDims(2))
                );
            // compute cell size
            double cellSize = (backCorner.at(0) - frontCorner.at(0)) / (double)gridDims(0);
            // create global grid
            grid = Cajita::createGlobalGrid
                (
                    comm,
                    partitioner,
                    periodicity,
                    frontCorner,
                    backCorner,
                    cellSize
                );
            // create layout on global grid
            // four degrees of freedom per grid point:
            // 0: real input value
            // 1: imag input value
            // 2: real part FFT
            // 3: imag part FFT
            layout = Cajita::createArrayLayout
                (
                    grid,
                    halo,
                    4,
                    Cajita::Cell()
                );
            array = Cajita::createArray<double, MemorySpace>( "internalFFTArray", layout);
            // compute grid dimension
            auto indexSpace = layout->indexSpace(Cajita::Own(), Cajita::Global());
            for (auto i = 0; i < 3; ++i)
                gridDims(i) = indexSpace.max(i) - indexSpace.min(i);
        }

        /// constuctor to set the cell layout the data is distributed to
        /// @param layout cell layout pointer on which the data is stored
        /// @param halo the width of the halo region, required for Cajita
        CajitaFFTSolver(std::shared_ptr<Cajita::GlobalGrid> appGrid, int halo) :
            CajitaFFTSolver()
        { 
            // store grid
            grid = appGrid;
            // create layout on global grid
            // four degrees of freedom per grid point:
            // 0: real input value
            // 1: imag input value
            // 2: real part FFT
            // 3: imag part FFT
            layout = Cajita::createArrayLayout
                (
                    grid,
                    halo,
                    4,
                    Cajita::Cell()
                );
            array = Cajita::createArray<double, MemorySpace>( "internalFFTArray", layout);
            // compute grid dimension
            auto indexSpace = layout->indexSpace(Cajita::Own(), Cajita::Global());
            for (auto i = 0; i < 3; ++i)
                gridDims(i) = indexSpace.max(i) - indexSpace.min(i);
        }
        ~CajitaFFTSolver() = default;

        /// function to set the grid dimensions of the grid to be used in the Cajita FFT solver
        /// @param gx   number of grid pwoints in x-dimension
        /// @param gy   number of grid points in y-dimension
        /// @param gz   number of grid points in z-dimension
        void setGridDim(int, int, int);

        /// virtual function to bring particle charges to the grid
        /// @param positions array with grid positions
        /// @param charges array with charge positions
        void q2grid( Kokkos::View<double*, MemorySpace>, 
                     Kokkos::View<double*, MemorySpace> );

        virtual void forwardFFT() = 0;

        virtual void backwardFFT() = 0;

        virtual void grid2E( Kokkos::View<double*, MemorySpace> ) = 0;
        virtual void grid2F( Kokkos::View<double*, MemorySpace> ) = 0;

        std::shared_ptr<Cajita::Array<double, Cajita::Cell, MemorySpace>> getResultArray()
        {
            return array;
        }

        std::shared_ptr<Cajita::ArrayLayout<Cajita::Cell>> getLocalLayout() { return layout; }

    protected:
        virtual Kokkos::View<int*, MemorySpace>& getGridDim() { return gridDims; }
        virtual std::shared_ptr<Cajita::GlobalGrid>& getGrid() { return grid; }
        virtual std::shared_ptr<Cajita::ArrayLayout<Cajita::Cell>>& getLayout() { return layout; }
        virtual std::shared_ptr<Cajita::Array<double, Cajita::Cell, MemorySpace>>& getArray() { return array; }       
        
    private:
        /// internal storage of the grid dimensions
        Kokkos::View<int*, MemorySpace> gridDims;
        /// internal grid to work on
        std::shared_ptr<Cajita::GlobalGrid> grid;
        /// internal pointer to the used array layout
        std::shared_ptr<Cajita::ArrayLayout<Cajita::Cell>> layout;
        /// internal data storeage (array)
        std::shared_ptr<Cajita::Array<double, Cajita::Cell, MemorySpace>> array;        
};

template< class ExecutionSpace, class MemorySpace > 
void CajitaFFTSolver<ExecutionSpace, MemorySpace>::q2grid
(
    Kokkos::View<double*, MemorySpace> positions,
    Kokkos::View<double*, MemorySpace> charges
)
{
    // bring charges to grid
    auto n_particles = charges.extent(0);

    auto grid = this->getGrid();
    auto layout = this->getLayout();
    auto array = this->getArray();
    auto view = array->view();
    auto cellSize = grid->cellSize();


    auto indexSpace = layout->indexSpace(Cajita::Own(), Cajita::Global());
    for (auto i = 0; i < n_particles; ++i)
    {
        int ix = (positions(3 * i + 0) - indexSpace.min(0) * cellSize) / cellSize;
        int iy = (positions(3 * i + 1) - indexSpace.min(1) * cellSize) / cellSize;
        int iz = (positions(3 * i + 2) - indexSpace.min(2) * cellSize) / cellSize;

        view(ix, iy, iz, 0) += charges(i);
        view(ix, iy, iz, 1) = 0.0;
    }

    std::cout << "charges brought to grid" << std::endl;
}

template< class ExecutionSpace, class MemorySpace >
std::ostream& operator<< (std::ostream& os, CajitaFFTSolver<ExecutionSpace, MemorySpace>& obj)
{
    auto array = obj.getResultArray();
    auto view = array->view();

    std::cerr << view.extent(0) << " - "
              << view.extent(1) << " - "
              << view.extent(2) << " - "
              << view.extent(3) << std::endl;

    for (int iz = 0; iz < view.extent(2); ++iz)
        for (int iy = 0; iy < view.extent(1); ++iy)
            for (int ix = 0; ix < view.extent(0); ++ix)
            {
                os << ix << " " << iy << " " << iz << ": ";
                for (int idx = 0; idx < view.extent(3); ++idx)
                    os << view(ix, iy, iz, idx) << " ";
                os << '\n';
            }
    return os;
}

#endif
