import os
import numpy as np
import pyfftw
from tqdm import tqdm
# from numba import njit, prange, set_num_threads

class DensityField3D():
    """
    Class for manipulating three-dimensional density fields and measuring power spectra and bispectra.

    Parameters
    ----------
    BoxSize : float
        Size of the periodic box in Mpc/h units.
    grid : int
        Number of grid points per dimension.
    delta_r : ndarray, optional
        Real-space density field to load and FFT. Shape must match the grid.
    delta_c : ndarray, optional
        Complex Fourier-space density field to load and inverse FFT. Shape must match the grid.
    n_threads : int, optional
        Number of threads to use for parallelization.
    FFTW_WISDOM : bool, optional
        Whether to use precomputed FFTW wisdom.
    r_dtype : str or np.dtype, optional
        Data type for the density field in real space. Must be a valid numpy data type.

    Attributes
    ----------
    BoxSize : float
        Size of the periodic box in Mpc/h units.
    grid : int
        Number of grid points per dimension.
    cell_size : float
        Physical size of a single grid cell.
    kF : float
        Fundamental mode of the box.
    kNyq : float
        Nyquist frequency of the grid.
    n_threads : int
        Number of threads used for parallelization.
    FFTW_WISDOM : bool
        Whether to use precomputed FFTW wisdom.
    r_dtype : str
        String representation of the data type for the density field in real space.
    c_dtype : str
        String representation of the data type for the density field in Fourier space.
    FFTW_FLAG : str
        String representation of the FFTW flag used.
    rshape : ndarray
        Shape of the real-space density grid.
    cshape : ndarray
        Shape of the Fourier-space density grid.
    r_fftgrid : ndarray
        Real-space density grid.
    c_fftgrid : ndarray
        Fourier-space density grid.
    fft_r2c : pyfftw.FFTW
        FFTW plan for real-to-complex Fourier transform.
    fft_c2r : pyfftw.FFTW
        FFTW plan for complex-to-real inverse Fourier transform.
    r_delta : ndarray
        Real-space density field.
    c_delta : ndarray
        Complex Fourier-space density field.
    kx : ndarray
        Array of x components of the wave vectors in Fourier space.
    ky : ndarray
        Array of y components of the wave vectors in Fourier space.
    kz : ndarray
        Array of z components of the wave vectors in Fourier space.
    kmesh : tuple of ndarrays
        Tuple of 3 ndarrays containing the x, y, and z components of the wave vectors in Fourier space.
    kgrid : ndarray
        Magnitude of the wave vectors in Fourier space.

    Methods
    -------
    compensate_MAS(MAS)
        Compensate for the mass assignment scheme used.
    Load_r2c(delta_r, MAS=None)
        Load a real-space density field and transform it to Fourier space.
    Load_c2r(delta_c)
        Load a complex Fourier-space density field and transform it to real space.
    """
    
    def __init__(self, BoxSize, grid, delta_r=None, delta_c=None, n_threads=1, FFTW_WISDOM=False, r_dtype=np.float32):
        assert grid%2 == 0, "choose an even grid size"
        
        self.BoxSize = BoxSize 
        self.grid = grid 
        self.cell_size = self.BoxSize/self.grid 
        self.kF = 2*np.pi / self.BoxSize
        self.kNyq = self.kF * self.grid / 2
        
        self.n_threads = n_threads
        self.FFTW_WISDOM = FFTW_WISDOM
        
        #Numerical precision either in np.floatXX or 'floatXX' format
        self.r_dtype = r_dtype
        if self.r_dtype == np.float32 or self.r_dtype == 'float32':
            self.r_dtype = 'float32'
            self.c_dtype = 'complex64'
        elif self.r_dtype == np.float64 or self.r_dtype == 'float64':
            self.r_dtype = 'float64'
            self.c_dtype = 'complex128'
        
        if self.FFTW_WISDOM:
            _make_FFTW_wisdom_3D(self.grid,self.r_dtype,self.c_dtype,self.n_threads)
            self.FFTW_FLAG="FFTW_WISDOM_ONLY"
        else:
            self.FFTW_FLAG="FFTW_ESTIMATE"
        
        #Setup FFTW grids and transforms
        self.rshape = np.array([self.grid,self.grid,self.grid])
        self.cshape = np.array([self.grid,self.grid,self.grid//2 + 1])
        self.r_fftgrid = pyfftw.empty_aligned(self.rshape, dtype=self.r_dtype)
        self.c_fftgrid = pyfftw.empty_aligned(self.cshape, dtype=self.c_dtype)
        
        self.fft_r2c = pyfftw.FFTW(self.r_fftgrid, self.c_fftgrid, axes=tuple(range(3)),\
                                direction="FFTW_FORWARD",threads=self.n_threads,flags=[self.FFTW_FLAG])
        
        self.fft_c2r = pyfftw.FFTW(self.c_fftgrid, self.r_fftgrid, axes=tuple(range(3)),\
                                direction="FFTW_BACKWARD",threads=self.n_threads,flags=[self.FFTW_FLAG])
        
        #Allocate array for real and complex grids to be saved and in case given load density and FFT
        self.r_delta = np.zeros_like(self.r_fftgrid)
        self.c_delta = np.zeros_like(self.c_fftgrid)
        
        #Setup mesh and k-space grid
        self.kx = 2 * np.pi * np.fft.fftfreq(self.grid, self.cell_size)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.grid, self.cell_size)
        self.kz = 2 * np.pi * np.fft.rfftfreq(self.grid, self.cell_size) # Note rfft.
        self.kmesh = np.meshgrid(self.kx,self.ky,self.kz,indexing="ij")
        self.kgrid = np.sqrt(self.kmesh[0]**2 + self.kmesh[1]**2 + self.kmesh[2]**2)
        
        #Set numba threads for parallelization
        # set_num_threads(n_threads)
        
    def compensate_MAS(self,MAS):
        """
        Apply the compensation for the chosen mode assignment scheme to the density field.

        Parameters
        ----------
        self : object
            Object that holds the required input parameters and intermediate results.
        MAS : str
            Mode assignment scheme to be used. Currently supported values are 'NGP' (nearest grid point),
            'CIC' (cloud-in-cell), 'TSC' (triangular-shaped cloud), and 'PCS' (piecewise cubic spline).

        Raises
        ------
        Exception
            If the specified mode assignment scheme is not one of the supported values.

        Returns
        -------
        None

        Notes
        -----
        This function applies the mode assignment compensation factor to the density field, according to the
        chosen scheme. The compensation factor corrects for the effect of the finite grid resolution on the
        measured power spectrum.

        """
        if   MAS=='NGP': p=1.
        elif MAS=='CIC': p=2.
        elif MAS=='TSC': p=3.
        elif MAS=='PCS': p=4.
        else:
            raise Exception(f"MAS {MAS} not implemented, options are NGP, CIC, TSC and PCS") 
        for i in range(3):
            fac = np.pi * self.kmesh[i]/self.kF/self.grid
            fac[fac==0.0]=1.
            mas_fac = (fac/np.sin(fac))**p
            mas_fac[fac==1.]=1.
            self.c_delta *= mas_fac
        
    def Load_r2c(self,delta_r,MAS=None):
        assert (delta_r.shape == self.r_delta.shape), "mismatching grid sizes"
        self.r_delta = delta_r.copy()
        np.copyto(self.r_fftgrid,self.r_delta)
        self.fft_r2c()
        self.c_delta = self.c_fftgrid.copy()
        if MAS!=None:
            self.compensate_MAS(MAS)
        
    def Load_c2r(self,delta_c,MAS=None):
        assert (delta_c.shape == self.c_delta.shape), "mismatching grid sizes"
        self.c_delta = delta_c.copy()
        if MAS!=None:
            self.compensate_MAS(MAS)
        np.copyto(self.c_fftgrid,self.c_delta)
        self.fft_c2r()
        self.r_delta = self.r_fftgrid.copy()
        self.c2r()
        
    def Pk(self,kmax_n=None):
        """
        Computes the binned power spectrum of the complex field using a loop.

        Parameters
        ----------
        kmax_n : int or None, optional
            Maximum wavenumber multiple of `self.kF` to consider for the power spectrum.
            If None (default), use `self.grid/2` as the maximum.

        Returns
        -------
        pk_array : numpy.ndarray
            Array of shape `(n_bins, 3)` containing the binned power spectrum values.
            Each row represents a bin and contains the mean wavenumber, power spectrum, 
            and number of modes in that bin, respectively.

        Notes
        -----
        This function computes the power spectrum of the complex field by looping
        over the grid and binning the values in wavenumber space. The width of each bin
        is 1 * `self.kF`, up to the maximum wavenumber specified by `kmax_n`. The power
        spectrum is normalized by the number of modes in each bin and by the volume
        of the simulation box.

        """

        if kmax_n==None:
            kbins = np.arange(1,self.grid/2+1)*self.kF
        else:
            kbins = np.arange(1,kmax_n+1)*self.kF

        norm, _ = np.histogram(self.kgrid,kbins,weights=np.ones_like(self.kgrid))        
        kmean, _ = np.histogram(self.kgrid,kbins,weights=self.kgrid)
        pk, kk = np.histogram(self.kgrid,kbins,weights=(self.c_delta.real**2 + self.c_delta.imag**2))
        kmean/= norm
        pk *= self.BoxSize**3 / self.grid**6 / norm
        return np.array([kmean, pk, norm]).T
        
    # def Pk(self):
    #     # return _Pk_numba(self.c_delta,self.kgrid,self.BoxSize,self.grid)


    def mask_c2r(self,delta_c,k_low,k_high):
        # Cut off complex density field and return the real space density field

        np.copyto(self.c_fftgrid,delta_c)
        self.c_fftgrid[self.kgrid >= k_high] = 0.+0.j
        self.c_fftgrid[self.kgrid < k_low] = 0.+0.j
        self.fft_c2r()
        return self.r_fftgrid
    
    def _Bk_counts(self,fc,dk,NBmax,triangle_type,verbose):
        # Helperfunction that computes the triangle counts/volume to normalize binned bispectrum measurements
        # For a given binning this only has to be computed ones and can be saved to disk to save time
        
        file_name = f"FFTest3D_BkCounts_LBox{self.BoxSize}_Grid{self.grid}_Binning{dk}kF_fc{fc}_NBins{NBmax}_TriangleType{triangle_type}.npy"
        
        if os.path.exists(file_name):
            if verbose: print(f"Loading Counts from {file_name}")
            counts = np.load(file_name,allow_pickle=True).item()
            if verbose: print(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")
            return counts

        counts = {}
        counts['counts_P'] = np.zeros(NBmax)

        if triangle_type=='All':
            counts['bin_centers'] = np.array([(i,j,l)\
                                              for i in fc+np.arange(0, (NBmax))*dk \
                                              for j in np.arange(fc, i+1, dk)\
                                              for l in np.arange(fc, j+1, dk) if i<=j+l+dk]) 
                                                #the +dk allows for open bins (Biagetti '21)

        elif triangle_type=='Squeezed':
            counts['bin_centers'] = np.array([(i,i,j)\
                                              for ji,j in enumerate(fc + np.arange(0,NBmax)*dk)\
                                              for i in fc+np.arange(ji+1,NBmax)*dk])
        elif triangle_type=='Equilateral':
            counts['bin_centers'] = np.array([(i,i,i)\
                                              for i in fc+np.arange(0,NBmax)*dk])
        if verbose: print(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")

        if verbose: print(f"Creating Grids for Counts...")
        c_ones = np.ones_like(self.c_fftgrid)
        r_ones_shells = np.zeros((NBmax,self.rshape[0],self.rshape[1],self.rshape[2]),dtype=self.r_dtype)
        for i in tqdm(range(NBmax),disable= not verbose):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high= self.kF * (fc + dk * i + dk/2)
            # print(k_low,k_high)
            r_ones_shells[i] = self.mask_c2r(c_ones,k_low,k_high)
            
        if verbose: print("Computing Powerspectrum Counts...",end=' ')
        counts['counts_P'] = _Pk_shells(r_ones_shells) * self.grid**3

        if verbose: print("Computing Triangle Counts...",end=' ')
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
        counts['counts_B'] = _Bk_shells(r_ones_shells, bin_indices) * self.grid**6

        np.save(file_name,counts)
        if verbose: print(f"Saved Triangle Counts to {file_name}")
                
        return counts
        
    def Bk(self,fc,dk,NBmax,triangle_type='All',verbose=False):
        """
        Computes binned bispectrum of field for given binning and triangles

        Parameters:
        -----------
        fc: float
            Center of first bin in units of the fundamental mode.
        dk: float
            Width of the bin in units of the fundamental mode.
        NBmax: int
            Total number of momentum bins such that bins are given by kf*[(fc + i)±dk/2 for i in range(NBmax)].
        triangle_type: str, optional (default='All')
            Type of triangles to include in the bispectrum calculation. 
            Options: 'All' (include all shapes of triangles), 'Squeezed' (only triangles k_1 > k_2 = k_3), 
            'Equilateral' (include only triangles k_1 = k_2 = k_3).
        verbose: bool, optional (default=False)
            If True, print progress statements.

        Returns:
        --------
        result: numpy.ndarray
            An array of shape (len(counts['bin_centers']),8) containing the bispectrum and related information.
            The columns contain: bin centers, P(k1), P(k2), P(k3), B(k1,k2,k3), counts_B.
            
        Notes:
        --------
        The first time the computation for a certain binning is being done, 
        this function will first compute the necessary mode counts for power spectrum and bispectrum normalization. 
        This is saved in a file in the local directory for later use, when measuring from other density fields but with the same binning.
        """
        
        counts = self._Bk_counts(fc,dk,NBmax,triangle_type,verbose)
        # return 0
        r_delta_shells = np.zeros((NBmax,self.rshape[0],self.rshape[1],self.rshape[2]),dtype=self.r_dtype)

        if verbose: print(f"Creating Grids for Measurements...")
        for i in tqdm(range(NBmax),disable= not verbose):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high= self.kF * (fc + dk * i + dk/2)
            r_delta_shells[i] = self.mask_c2r(self.c_delta,k_low,k_high)

        if verbose: print(f"Computing Powerspectrum...",end=' ') 
        P = _Pk_shells(r_delta_shells) * self.BoxSize**3 / counts['counts_P'] / self.grid**3

        if verbose: print(f"Computing Bispectrum...",end=' ')    
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
        B = _Bk_shells(r_delta_shells,bin_indices) * self.BoxSize**6 / self.grid**3

        result = np.ones((len(counts['bin_centers']),8))
        result[:,:3] = counts['bin_centers']
        result[:,3:6] = P[bin_indices]
        result[:,6] = B/counts['counts_B']
        result[:,7] = counts['counts_B']

        return result

def _Pk_shells(r_delta_shells):
    # Helperfunction to compute powerspectrum of binned real density fields
        P_measured = np.zeros(len(r_delta_shells))
        for i in tqdm(range(len(r_delta_shells))):
            P_measured[i] = np.sum((r_delta_shells[i]**2))
        return P_measured
    
def _Bk_shells(r_delta_shells, bin_indices):
    # Helperfunction to compute bispectrum of binned real density fields
        B_measured = np.zeros(len(bin_indices))
        for bin_i in tqdm(range(len(bin_indices))):
            bin_index = bin_indices[bin_i]
            B_measured[bin_i] = np.sum(r_delta_shells[bin_index[0]]\
                                        *r_delta_shells[bin_index[1]]\
                                        *r_delta_shells[bin_index[2]])
        return B_measured


def _make_FFTW_wisdom_3D(grid,r_dtype,c_dtype,nthreads):
    pyfftw.forget_wisdom()
    file_name = f"pyFFTW_3D_grid{grid}_dtype{r_dtype}_threads{nthreads}.npy"
    if os.path.exists(file_name):
        # print(f"Loaded FFT Wisdom from {file_name}")
        pyfftw.import_wisdom(np.load(file_name))
    else:
        # print(f"Computing FFT Wisdom and Saving to {file_name}")
        rshape = np.array([grid,grid,grid])
        cshape = rshape.copy()
        cshape[-1] = (cshape[-1] // 2) + 1
        r_fftgrid = pyfftw.empty_aligned(rshape, dtype=r_dtype)
        c_fftgrid = pyfftw.empty_aligned(cshape, dtype=c_dtype)
        fft = pyfftw.FFTW(r_fftgrid, c_fftgrid,axes=tuple(range(3)), direction="FFTW_FORWARD",threads=nthreads,flags=['FFTW_MEASURE'])
        inv_fft = pyfftw.FFTW(c_fftgrid, r_fftgrid,axes=tuple(range(3)), direction="FFTW_BACKWARD",threads=nthreads,flags=['FFTW_MEASURE'])
        np.save(file_name,pyfftw.export_wisdom())
        
# @njit
# def _Pk_numba(c_delta,kgrid,BoxSize,grid):
#     # Computes binned powerspectrum using a loop over complex field
#     # in bins of width 1*kF up to kmax or Nyquist frequency

#     cell_size = BoxSize/grid
#     kF = 2*np.pi / BoxSize
#     kNyq = kF * grid / 2
#     kmax = kNyq

#     kmax_n = np.int64((np.ceil(kmax/kF)))    
#     Pks = np.zeros((kmax_n-1,3),dtype=np.float64)

#     for kxi in range(c_delta.shape[0]):
#         for kyi in range(c_delta.shape[1]):
#             for kzi in range(c_delta.shape[2]):
#                 if kxi==0 and kyi==0 and kzi==0: continue

#                 k = kgrid[kxi,kyi,kzi]
#                 if k > kmax: continue
#                 k_index = np.int64((np.floor(k/kF)))

#                 delta_r = c_delta[kxi,kyi,kzi].real
#                 delta_i = c_delta[kxi,kyi,kzi].imag
#                 delta2 = delta_r**2 + delta_i**2

#                 Pks[k_index-1,0] += k
#                 Pks[k_index-1,1] += delta2
#                 Pks[k_index-1,2] += 1.

#     Pks[:,0] /= Pks[:,2]
#     Pks[:,1] *= 1/Pks[:,2] * BoxSize**3 / grid**6
#     return Pks

def PkX(r_delta_1,BoxSize,r_delta_2=None,MAS=[None,None],n_threads=1):
    """
    Computes the cross-powerspectrum of given real density fields of size BoxSize.

    Parameters:
    -----------
    r_delta_1 : numpy.ndarray
        Real density field of shape (grid, grid, grid) and dtype float32 or float64.
    BoxSize : float
        Size of the simulation box.
    r_delta_2 : numpy.ndarray, optional (default=None)
        Real density field of shape (grid, grid, grid) and dtype float32 or float64.
    MAS : list of 2 str, optional (default=[None,None])
        Mass assignment scheme (MAS) for each density field. Possible values are: 
        'NGP' (nearest grid point), 'CIC' (cloud in cell), 'TSC' (triangular shaped cloud), 
        'PCS' (piecewise cubic spline), or None (no MAS correction applied).
    n_threads : int, optional (default=1)
        Number of threads to use in FFTW computation.
    
    Returns:
    --------
    tuple of 3 or 4 numpy.ndarray
        Tuple containing the following arrays:
        kmean : numpy.ndarray
            Mean k of each k-bin in the power spectrum.
        pk_1 : numpy.ndarray
            Self-power spectrum of r_delta_1.
        norm : numpy.ndarray
            Mode count of each k-bin.
        pk_2 : numpy.ndarray, optional
            Self-power spectrum of r_delta_2, returned only if r_delta_2 is not None.
        pk_X : numpy.ndarray, optional
            Cross-power spectrum between r_delta_1 and r_delta_2, returned only if r_delta_2 is not None.
    
    """
        
    grid = r_delta_1.shape[0]
    cell_size = BoxSize/grid
    kF = 2*np.pi/BoxSize
    
    kx = 2 * np.pi * np.fft.fftfreq(grid, cell_size)
    ky = 2 * np.pi * np.fft.fftfreq(grid, cell_size)
    kz = 2 * np.pi * np.fft.rfftfreq(grid, cell_size)
    k = np.meshgrid(kx,ky,kz,indexing="ij")
    kgrid = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
    
    rshape = np.array([grid,grid,grid])
    cshape = np.array([grid,grid,grid//2+1])
    
    r_dtype = r_delta_1.dtype
    if r_dtype == np.float32 or r_dtype == 'float32':
        r_dtype = 'float32'
        c_dtype = 'complex64'
    elif r_dtype == np.float64 or r_dtype == 'float64':
        r_dtype = 'float64'
        c_dtype = 'complex128'
    
    r_fftgrid = pyfftw.empty_aligned(rshape, dtype=r_dtype)
    c_fftgrid = pyfftw.empty_aligned(cshape, dtype=c_dtype)
    fft = pyfftw.FFTW(r_fftgrid, c_fftgrid, axes=tuple(range(3)), direction="FFTW_FORWARD",threads=n_threads,flags=['FFTW_ESTIMATE'])
    inv_fft = pyfftw.FFTW(c_fftgrid, r_fftgrid, axes=tuple(range(3)), direction="FFTW_BACKWARD",threads=n_threads,flags=['FFTW_ESTIMATE'])
    
    def compensate_MAS(c_delta,MAS):
        p = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in range(3):
            fac = np.pi * k[i]/kF/grid
            fac[fac==0.0]=1.
            mas_fac = (fac/np.sin(fac))**p
            mas_fac[fac==1.]=1.
            c_delta *= mas_fac
            return c_delta 
    
    np.copyto(r_fftgrid,r_delta_1)
    fft()
    c_delta_1 = c_fftgrid.copy()
    if MAS[0]!=None: c_delta_1 = compensate_MAS(c_delta_1,MAS[0])

    kbins = np.arange(1,grid/2+1)*kF

    # Compute the mode counts
    norm, _ = np.histogram(kgrid,kbins,weights=np.ones_like(kgrid))   
    
    # Compute the mean k of the bin
    kmean, _ = np.histogram(kgrid,kbins,weights=kgrid)
    kmean /= norm
    
    # Compute the self-powerspectrum of the density field
    pk_1, _ = np.histogram(kgrid,kbins,weights=(c_delta_1.real**2 + c_delta_1.imag**2))
    pk_1 *= BoxSize**3 / grid**6 / norm
    
    if type(r_delta_2)!= type(None):
        assert r_delta_1.shape[0] == r_delta_2.shape[1], "Only equal dimensions are supported"
        np.copyto(r_fftgrid,r_delta_2)
        fft()
        c_delta_2 = c_fftgrid.copy()
        if MAS[1]!=None: c_delta_2 = compensate_MAS(c_delta_2,MAS[1])
    
        pk_2, _ = np.histogram(kgrid,kbins,weights=(c_delta_2.real**2 + c_delta_2.imag**2))
        pk_X, _ = np.histogram(kgrid,kbins,weights=(c_delta_1 *  + np.conj(c_delta_2)))
    

        pk_2 *= BoxSize**3 / grid**6 / norm
        pk_X = np.real(pk_X * BoxSize**3 / grid**6 / norm)
        
        return np.array([kmean, pk_1, pk_2, pk_X, norm]).T
    
    return np.array([kmean, pk_1, norm]).T

