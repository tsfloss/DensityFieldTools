# DensityFieldTools
Tools for manipulating density fields and measuring power spectra and bispectra

to do:
    - currently parallelization/n_threads>1 only affects the FFT, it would be nice to parallelize the bispectrum measurement
            (a simple numba implementation suffers from poor floating point performance when used on float32 grids, leading to inaccurate measurements)

dependencies:
    - numpy
    - pyfftw
    - tqdm
    - matplotlib (for the example notebook)
    
