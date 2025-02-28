# Synchrotron PDF Computation

## Overview

This repository contains Python scripts for computing synchrotron emission properties using MPI parallelization. The main script, `synchrotron_pdf.py`, utilizes functions and classes defined in `synchrotron.py` to analyze and process magnetic field and plasma density data from simulation outputs.

## Files

- **synchrotron_pdf.py**: This script is responsible for orchestrating the synchrotron emission calculations by:
  - Initializing MPI communicators and setting up the computational grid.
  - Loading simulation data and distributing it efficiently across multiple processors.
  - Computing synchrotron emission properties, such as Stokes parameters (I, Q, U), Faraday depth, and polarization intensities.
  - Generating output files with probability density functions (PDFs) and integral scale calculations.
  
- **synchrotron.py**: This module provides essential functions and classes to handle parallel computations and synchrotron emission calculations efficiently. It includes:
  - `create_grid_beam`: Sets up a structured Cartesian topology of MPI processes in up to three dimensions and assigns separate communicators for parallel computing.
  - `create_mem_window`: Implements shared memory windows for efficient data sharing among MPI processes without redundant copies, optimizing RAM usage.
  - `sum_beam`: A helper function that sums data along a specified beam direction using MPI reduction operations.
  - `Synchrotron` Class: Handles core physical computations related to synchrotron emission. It includes methods to:
    - Set global class variables such as electron density exponent, frequency, and magnetic field scaling.
    - Compute Stokes parameters (I, Q, U) based on magnetic field configurations and plasma density.
    - Calculate Faraday depth and polarization properties for given lines of sight.

## Dependencies

Ensure you have the following Python libraries installed:

- `numpy`
- `mpi4py`
- `matplotlib`
- `scipy`
- `h5py`
- `pencil`

## How to Run

1. Ensure MPI is properly installed on your system.
2. Run the script using `mpirun` or `mpiexec` with the desired number of processors:
   ```bash
   mpirun -np <num_processors> python synchrotron_pdf.py
   ```

## Functionality

### synchrotron_pdf.py

- Initializes MPI processes and assigns computational tasks.
- Loads simulation data and distributes it across processes.
- Computes:
  - Synchrotron emissivity
  - Faraday depth
  - Stokes parameters (I, Q, U)
  - Fractional polarization
  - Power spectra of polarization intensity
  - PDFs of various quantities
- Saves computed results to output files.

### synchrotron.py

- Defines `Synchrotron` class for physical computations.
- Implements MPI functions for efficient parallel processing.
- Provides utilities for data sharing across processors.
- Handles grid creation, memory optimization, and collective data operations.

## Output

Results are stored in the `pdf_data` directory, including:

- `pdf_FD_*.txt`: Faraday depth probability distributions.
- `pdf_Sync_*.txt`: PDFs of Stokes parameters.
- `avg_map_*.h5`: HDF5 files with averaged intensity and polarization maps.
- `intscale_PI_FP.txt`: Integral scale values for polarized emission.

## License

This project is open-source. Feel free to use and modify it as needed.

## Author

Developed by Naveen Jingade in collaboration with Sharanya Sur at IIA, Bangalore. For questions, contact naveenjingade@gmail.com.

