import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import ducc0
from tqdm import tqdm
from numba import njit
from tqdm.contrib.concurrent import thread_map

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
    nthreads : int, optional
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
    nthreads : int
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
    delta_r : ndarray
        Real-space density field.
    delta_c : ndarray
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
    _compensate_MAS(MAS)
        Compensate for the mass assignment scheme used.
    Load_r2c(delta_r, MAS=None)
        Load a real-space density field and transform it to Fourier space.
    Load_c2r(delta_c)
        Load a complex Fourier-space density field and transform it to real space.
    """
    
    def __init__(self, BoxSize, grid, nthreads=1):
        assert grid%2 == 0, "choose an even grid size"
        
        self.BoxSize = BoxSize 
        self.grid = grid 
        self.cell_size = self.BoxSize/self.grid 
        self.kF = 2*np.pi / self.BoxSize
        self.kNyq = self.kF * self.grid / 2
        
        self.nthreads = nthreads
        
        # # Setup mesh and k-space grid
        self.kx = 2 * np.pi * np.fft.fftfreq(self.grid, self.cell_size)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.grid, self.cell_size)
        self.kz = 2 * np.pi * np.fft.rfftfreq(self.grid, self.cell_size) # Note rfft.
        self.kmesh = np.array(np.meshgrid(self.kx,self.ky,self.kz,indexing="ij"))
        self.kgrid = np.sqrt(self.kmesh[0]**2 + self.kmesh[1]**2 + self.kmesh[2]**2)
        
    def _compensate_MAS(self,MAS):
        """
        Apply the compensation for the chosen mass assignment scheme to the density field.
        """
        p = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in range(3):
            fac = np.pi * self.kmesh[i]/self.kF/self.grid
            fac[fac==0.0]=1.
            mas_fac = (fac/np.sin(fac))**p
            mas_fac[fac==1.]=1.
            self.delta_c *= mas_fac
        
    def read_real(self,delta_r,MAS=None):
        assert (delta_r.shape == (self.grid,self.grid,self.grid)), "mismatching grid sizes"
        self.delta_r = delta_r
        self.delta_c = ducc0.fft.r2c(self.delta_r,nthreads=self.nthreads)
        if MAS!=None:
            self._compensate_MAS(MAS)
        
    def read_complex(self,delta_c,MAS=None):
        assert (delta_c.shape == self.delta_c.shape), "mismatching grid sizes"
        self.delta_c = delta_c
        if MAS!=None:
            self._compensate_MAS(MAS)
        self.delta_r = ducc0.fft.c2r(self.delta_c,nthreads=self.nthreads,lastsize=self.grid,forward=False,inorm=2)
        
    def Pk(self):
        return _Pk_numba(self.delta_c,self.BoxSize)

    def bin_c2r(self,delta_c,k_low,k_high):
        """
        Bin complex density field and return the real space density field
        """
        delta_c[self.kgrid >= k_high] = 0.+0.j
        delta_c[self.kgrid < k_low] = 0.+0.j
        return ducc0.fft.c2r(delta_c,nthreads=self.nthreads,lastsize=self.grid,forward=False,inorm=2)
    
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
        c_ones = np.ones_like(self.delta_c)
        r_ones_shells = np.zeros((NBmax,*self.delta_r.shape),dtype=self.delta_r.dtype)
        for i in tqdm(range(NBmax),disable= not verbose):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high= self.kF * (fc + dk * i + dk/2)
            r_ones_shells[i] = self.bin_c2r(c_ones.copy(),k_low,k_high)
            
        if verbose: print("Computing Powerspectrum Counts...",end=' ')
        counts['counts_P'] = np.array(thread_map(lambda bin_i: np.sum(r_ones_shells[bin_i]**2)\
                                ,np.arange(len(r_ones_shells)),max_workers=self.nthreads,tqdm_class=tqdm,disable= not verbose)) * self.grid**3

        if verbose: print("Computing Triangle Counts...",end=' ')
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)

        counts['counts_B'] = B = np.array(thread_map(lambda bin_i: np.sum(r_ones_shells[bin_indices[bin_i][0]]*r_ones_shells[bin_indices[bin_i][1]]*r_ones_shells[bin_indices[bin_i][2]])\
                                ,np.arange(len(bin_indices)),max_workers=self.nthreads,tqdm_class=tqdm,disable= not verbose)) * self.grid**6

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
        delta_r_shells = np.zeros((NBmax,*self.delta_r.shape),dtype=self.delta_r.dtype)

        if verbose: print(f"Creating Grids for Measurements...")
        for i in tqdm(range(NBmax),disable= not verbose):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high= self.kF * (fc + dk * i + dk/2)
            delta_r_shells[i] = self.bin_c2r(self.delta_c.copy(),k_low,k_high)
            
        if verbose: print(f"Computing Powerspectrum...",end=' ')
        P = np.array(thread_map(lambda bin_i: np.sum(delta_r_shells[bin_i]**2)\
                                ,np.arange(len(delta_r_shells)),max_workers=self.nthreads,tqdm_class=tqdm,disable= not verbose)) * self.BoxSize**3 / counts['counts_P'] / self.grid**3

        if verbose: print(f"Computing Bispectrum...",end=' ')    
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
        B = np.array(thread_map(lambda bin_i: np.sum(delta_r_shells[bin_indices[bin_i][0]]*delta_r_shells[bin_indices[bin_i][1]]*delta_r_shells[bin_indices[bin_i][2]])\
                                ,np.arange(len(bin_indices)),max_workers=self.nthreads,tqdm_class=tqdm,disable= not verbose)) * self.BoxSize**6 / self.grid**3

        result = np.ones((len(counts['bin_centers']),8))
        result[:,:3] = counts['bin_centers']
        result[:,3:6] = P[bin_indices]
        result[:,6] = B/counts['counts_B']
        result[:,7] = counts['counts_B']

        return result

@njit
def _Pk_numba(delta_c_1,BoxSize):
    """
    Computes the self-power spectrum of a complex fields delta_c_1
    in bins of width 1*kF up to kmax or Nyquist frequency.

    Parameters:
    -----------
    delta_c_1: ndarray of complex numbers with shape (N, N, N)
        The first complex field
    kgrid: ndarray of floats with shape (N, N, N)
        The grid of wavenumbers corresponding to the complex fields.
    BoxSize: float
        The size of the box containing the complex fields in units of Mpc/h.

    Returns:
    --------
    ndarray of floats with shape (kNyq/2-1, 3)
        The first column contains the k values.
        The second and third columns contain the power spectrum of delta_c_1 and delta_c_2 respectively.
        The fourth column contains the cross-power spectrum of delta_c_1 and delta_c_2.
        The last column contains the number of modes in each bin.
    """
    
    grid = delta_c_1.shape[0]
    cell_size = BoxSize/grid
    kF = 2*np.pi / BoxSize
    kNyq = grid // 2
    kmax = kNyq

    Pks = np.zeros((kmax,3),dtype=np.float64)

    for kxi in range(delta_c_1.shape[0]):
        kx = (kxi-grid if (kxi>grid//2) else kxi)
        for kyi in range(delta_c_1.shape[1]):
            ky = (kyi-grid if (kyi>grid//2) else kyi)
            for kzi in range(delta_c_1.shape[2]):
                kz = kzi

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==grid//2 and grid%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==grid//2 and grid%2==0):
                        if ky<0.0: continue

                k = np.sqrt(kx**2. + ky**2. + kzi**2.)
                if k >= kmax: continue
                k_index = np.int64(k)
                # print(k,k_index)
                
                delta_1 = delta_c_1[kxi,kyi,kzi]
                delta_1_norm2 = delta_1.real**2 + delta_1.imag**2

                Pks[k_index,0] += k
                Pks[k_index,1] += delta_1_norm2
                Pks[k_index,-1] += 1.

    Pks[:,0] *= kF / Pks[:,-1]
    Pks[:,1] *= 1/Pks[:,-1] * BoxSize**3 / grid**6

    Pks = Pks[1:]

    return Pks
      

@njit
def _PkX_numba(delta_c_1,delta_c_2,BoxSize):
    """
    Computes the self- and cross-power spectra of two complex fields delta_c_1 and delta_c_2
    in bins of width 1*kF up to kmax or Nyquist frequency.

    Parameters:
    -----------
    delta_c_1: ndarray of complex numbers with shape (N, N, N)
        The first complex field
    delta_c_2: ndarray of complex numbers with shape (N, N, N)
        The second complex field. Must have equal dimensions with delta_c_1.
    kgrid: ndarray of floats with shape (N, N, N)
        The grid of wavenumbers corresponding to the complex fields.
    BoxSize: float
        The size of the box containing the complex fields in units of Mpc/h.

    Returns:
    --------
    ndarray of floats with shape (kNyq/2-1, 5)
        The first column contains the k values.
        The second and third columns contain the power spectrum of delta_c_1 and delta_c_2 respectively.
        The fourth column contains the cross-power spectrum of delta_c_1 and delta_c_2.
        The last column contains the number of modes in each bin.
    """
    
    grid = delta_c_1.shape[0]
    cell_size = BoxSize/grid
    kF = 2*np.pi / BoxSize
    kNyq = grid // 2
    kmax = kNyq

    Pks = np.zeros((kmax,5),dtype=np.float64)

    for kxi in range(delta_c_1.shape[0]):
        kx = (kxi-grid if (kxi>grid//2) else kxi)
        for kyi in range(delta_c_1.shape[1]):
            ky = (kyi-grid if (kyi>grid//2) else kyi)
            for kzi in range(delta_c_1.shape[2]):
                kz = kzi

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==grid//2 and grid%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==grid//2 and grid%2==0):
                        if ky<0.0: continue

                k = np.sqrt(kx**2. + ky**2. + kzi**2.)
                if k >= kmax: continue
                k_index = np.int64(k)
                
                delta_1 = delta_c_1[kxi,kyi,kzi]
                delta_1_norm2 = delta_1.real**2 + delta_1.imag**2
                
                delta_2 = delta_c_2[kxi,kyi,kzi]
                delta_2_norm2 = delta_2.real**2 + delta_2.imag**2
                    
                delta_X_norm2 = (delta_1 * np.conj(delta_2)).real

                Pks[k_index,0] += k
                Pks[k_index,1] += delta_1_norm2
                Pks[k_index,2] += delta_2_norm2
                Pks[k_index,3] += delta_X_norm2
                Pks[k_index,-1] += 1.

    Pks[:,0] *= kF / Pks[:,-1]
    Pks[:,1] *= 1/Pks[:,-1] * BoxSize**3 / grid**6
    Pks[:,2] *= 1/Pks[:,-1] * BoxSize**3 / grid**6
    Pks[:,3] *= 1/Pks[:,-1] * BoxSize**3 / grid**6
    Pks = Pks[1:]
    return Pks

def PkX(delta_r_1,BoxSize,delta_r_2=None,MAS=[None,None],nthreads=1):
    """
    Computes the self- and cross-power spectra of two real fields delta_r_1 and delta_r_2
    in bins of width 1*kF up to kmax or Nyquist frequency.

    Parameters:
    -----------
    delta_r_1: ndarray of real numbers with shape (N, N, N)
        The first real field
    delta_r_2: ndarray of real numbers with shape (N, N, N)
        The second real field. Must have equal dimensions with delta_r_1.
    BoxSize: float
        The size of the box containing the real fields in units of Mpc/h.

    Returns:
    --------
    ndarray of floats with shape (kNyq/2-1, 3) if one real field is given
        The first column contains the k values.
        The second column contains the power spectrum of delta_r_1.
        The last column contains the number of modes in each bin.
    
    ndarray of floats with shape (kNyq/2-1, 5) if two real fields are given
        The first column contains the k values.
        The second and third columns contain the power spectrum of delta_r_1 and delta_r_2 respectively.
        The fourth column contains the cross-power spectrum of delta_r_1 and delta_r_2.
        The last column contains the number of modes in each bin.
    """
        
    grid = delta_r_1.shape[0]
    cell_size = BoxSize/grid
    kF = 2*np.pi/BoxSize
    
    kx = 2 * np.pi * np.fft.fftfreq(grid, cell_size)
    ky = 2 * np.pi * np.fft.fftfreq(grid, cell_size)
    kz = 2 * np.pi * np.fft.rfftfreq(grid, cell_size)
    k = np.meshgrid(kx,ky,kz,indexing="ij")
    kgrid = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)

    rshape = np.array([grid,grid,grid])
    cshape = np.array([grid,grid,grid//2+1])
    
    r_dtype = delta_r_1.dtype
    if r_dtype == np.float32 or r_dtype == 'float32':
        r_dtype = 'float32'
        c_dtype = 'complex64'
    elif r_dtype == np.float64 or r_dtype == 'float64':
        r_dtype = 'float64'
        c_dtype = 'complex128'

    r_fftgrid = pyfftw.empty_aligned(rshape, dtype=r_dtype)
    c_fftgrid = pyfftw.empty_aligned(cshape, dtype=c_dtype)
    fft = pyfftw.FFTW(r_fftgrid, c_fftgrid, axes=tuple(range(3)), direction="FFTW_FORWARD",threads=nthreads,flags=['FFTW_ESTIMATE'])
    inv_fft = pyfftw.FFTW(c_fftgrid, r_fftgrid, axes=tuple(range(3)), direction="FFTW_BACKWARD",threads=nthreads,flags=['FFTW_ESTIMATE'])
    
    def _compensate_MAS(delta_c,MAS):
        p = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in range(3):
            fac = np.pi * k[i]/kF/grid
            fac[fac==0.0]=1.
            mas_fac = (fac/np.sin(fac))**p
            mas_fac[fac==1.]=1.
            delta_c *= mas_fac
        return delta_c
    
    np.copyto(r_fftgrid,delta_r_1)
    fft()
    delta_c_1 = c_fftgrid.copy()
    if MAS[0]!=None: delta_c_1 = _compensate_MAS(delta_c_1,MAS[0])
    
    if type(delta_r_2)!= type(None):
        assert delta_r_1.shape[0] == delta_r_2.shape[1], "Only equal dimensions are supported"
        np.copyto(r_fftgrid,delta_r_2)
        fft()
        delta_c_2 = c_fftgrid.copy()
        if MAS[1]!=None: delta_c_2 = _compensate_MAS(delta_c_2,MAS[1])
        
        return _PkX_numba(delta_c_1,delta_c_2,BoxSize)
    else:
        return _Pk_numba(delta_c_1,BoxSize)