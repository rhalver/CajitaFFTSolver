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
        /// @param gx   number of grid pwoints in x-dimension
        CajitaFFTSolver(
                            MPI_Comm comm,
                            std::vector<int> gridDim,
                            std::vector<bool> periodicity,
                            std::vector<double> frontCorner,
                            std::vector<double> backCorner
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
            double cellSize = backCorner.at(0) - frontCorner.at(0) / (double)gridDims(0);
            grid = Cajita::createGlobalGrid
                (
                    comm,
                    partitioner,
                    periodicity,
                    frontCorner,
                    backCorner,
                    cellSize
                );
            layout = Cajita::createArrayLayout
                (
                    grid,
                    0,
                    1,
                    Cajita::Cell()
                );
            auto indexSpace = layout->indexSpace(Cajita::Own(), Cajita::Global());
            for (auto i = 0; i < 3; ++i)
                gridDims(i) = indexSpace.max(i) - indexSpace.min(i);
        }
        /// constuctor to set the cell layout the data is distributed to
        /// @param layout cell layout pointer on which the data is stored
        CajitaFFTSolver(std::shared_ptr<Cajita::GlobalGrid> appGrid) :
            CajitaFFTSolver()
        {  
            grid = appGrid;
            layout = Cajita::createArrayLayout
                (
                    grid,
                    0,
                    1,
                    Cajita::Cell()
                );
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
    private:
        /// internal storage of the grid dimensions
        Kokkos::View<int*, MemorySpace> gridDims;
        /// internal grid to work on
        std::shared_ptr<Cajita::GlobalGrid> grid;
        /// internal pointer to the used array layout
        std::shared_ptr<Cajita::ArrayLayout<Cajita::Cell>> layout;
};

#endif
