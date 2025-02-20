# Synchrotron PDF Computation

## Overview

This repository contains Python scripts for computing synchrotron emission properties using MPI parallelization. The main script, `synchrotron_pdf.py`, utilizes functions and classes defined in `synchrotron.py` to analyze and process magnetic field and plasma density data from simulation outputs.

## Files

-**synchrotron_pdf.py**: Main script that loads simulation data, initializes MPI processes, and computes various synchrotron emission properties such as Stokes parameters, Faraday depth, and polarization intensities.
- **synchrotron.py**: Contains helper functions and the `Synchrotron` class, which includes methods for setting up MPI communicators, creating shared memory windows, and computing necessary physical parameters.

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

### synchrotron\_pdf.py

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

## Output

Results are stored in the `pdf_data` directory, including:

- `pdf_FD_*.txt`: Faraday depth probability distributions.
- `pdf_Sync_*.txt`: PDFs of Stokes parameters.
- `avg_map_*.h5`: HDF5 files with averaged intensity and polarization maps.
- `intscale_PI_FP.txt`: Integral scale values for polarized emission.

## License

This project is open-source. Feel free to use and modify it as needed.

## Author

Developed by NaveenJingade. For questions, contact naveenjingade@gmail.com.

