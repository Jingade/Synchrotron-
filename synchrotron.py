import numpy as np
from mpi4py import MPI
import gc

def create_grid_beam(nprocx, nprocy, nprocz=None):
    """
    Creates the 2D or 3D cartesian grid of processors, given the number 
    of processors nprocx(along X-direction), nprocy(along Y-direction)
    nprocz(along Z-direction) and returns the index for processors 
    (ipx, ipy, ipz). It also creates the separate communicator in 
    x, y and z direction and label it as XBEAM_COMM, YBEAM_COMM
    and ZBEAM_COMM. It also creates addition planar processors
    if nprocz is present. 

    Parameters:
    - nprocx: number of processor in x-direction
    - nprocy: number of processor in y-direction
    - nprocy: number of processor in z-direction (optional)
	
    Return:
    ipx : processor index in x -direction
    ipy : processor index in y -direction
    ipz : processor index in z -direction

    XBEAM_COMM: communicator for processors group in x-direction (columns).
                There are nprocy number of XBEAM_COMM communicators  
    YBEAM_COMM: communicator for processors group in y-direction (rows).
                There are nprocx number of YBEAM_COMM communicators  
    """

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if nprocz is None:
        # Ensure the product of Y_BEAM and X_BEAM processors matches the total number of processors
        if nprocy * nprocx != size:
            raise ValueError("The product of Y_BEAM and X_BEAM processors must equal the total number of processors.")

        grid_dims = (nprocy, nprocx)
        # Create the Cartesian topology                                                                                                                                                                   
        cart_comm = comm.Create_cart(dims=grid_dims, periods=(False, False), reorder=False)

        # Get the Cartesian coordinates of each process
        ipy, ipx = cart_comm.Get_coords(rank)

        # Create communicator for X_BEAM
        XBEAM_COMM = comm.Split(color=ipy, key=ipx)

        # Create communicator for Y_BEAM
        YBEAM_COMM = comm.Split(color=ipx, key=ipy)

        return ipx, ipy, XBEAM_COMM, YBEAM_COMM

    else:
        # Ensure the product of X, Y, and Z processors matches the total number of processors
        if nprocy * nprocx * nprocz != size:
            raise ValueError("The product of X, Y, and Z processors must equal the total number of processors.")

        grid_dims = (nprocy, nprocx, nprocz)
        # Create the Cartesian topology                                                                                                                                                                   
        cart_comm = comm.Create_cart(dims=grid_dims, periods=(False, False, False), reorder=False)

        # Get the Cartesian coordinates of each process
        ipy, ipx, ipz = cart_comm.Get_coords(rank)

        # Create communicator for X_BEAM (same color for same Y and Z)
        XBEAM_COMM = comm.Split(color=ipy + nprocy * ipz, key=ipx)

        # Create communicator for Y_BEAM (same color for same X and Z)
        YBEAM_COMM = comm.Split(color=ipx + nprocx * ipz, key=ipy)

        # Create communicator for Z_BEAM (same color for same X and Y)
        ZBEAM_COMM = comm.Split(color=ipx + nprocx * ipy, key=ipz)

        # Create communicator for XY_PLANE (same Z)
        XYPLANE_COMM = comm.Split(color=ipz, key=ipx + nprocx * ipy)

        # Create communicator for XZ_PLANE (same Y)
        XZPLANE_COMM = comm.Split(color=ipy, key=ipx + nprocx * ipz)

        # Create communicator for YZ_PLANE (same X)
        YZPLANE_COMM = comm.Split(color=ipx, key=ipy + nprocy * ipz)

        return ipx, ipy, ipz, XBEAM_COMM, YBEAM_COMM, ZBEAM_COMM, YZPLANE_COMM, XZPLANE_COMM, XYPLANE_COMM

    

      ############################################################

def create_mem_window(lfirst_proc, array_count, N, color, COMM):
    """
    This function creates a shared memory window on the RAM for 
    the given 'color', only for those processors where lfirst_proc is TRUE. 
    Rest of the processors in the communicator COMM query that window
    using the pointer buf and access the data. This avoids unnecassry copying 
    of the data, if it is available on the RAM.

    lfirst_proc : any processor whose rank is 0 for the given communicator COMM
    array_count : numbers of arrays in the shared window
    N           : size of the array along one of the dimensions 
    color       : indices of the processors along which data window is needed
    COMM        : commuinicator through which other processors can access the data

    """

    win_dict   = {}
    data_size  = array_count * N**3
    array_size = N**3
    elem_size  = MPI.DOUBLE.Get_size()
    
    if lfirst_proc:
        # first a memory space is created for array_count number of 3D array having double precision
        # in bytes for each lfirst_proc processors.
        win_dict[color] = MPI.Win.Allocate_shared(data_size*elem_size, elem_size, comm = COMM)
    else:
        # other processors do not allocate the memory, they join the shared window
        win_dict[color] = MPI.Win.Allocate_shared(0, elem_size, comm = COMM)
        
    #Fence to synchronise processes before quering the window
    win_dict[color].Fence()
        
    #Query the shared memory buffer on all processesors
    buf, item_size = win_dict[color].Shared_query(0) # rank 0 in XZPLANE communicators                                                                                                                           

    # create the numpy array like reference to the memory window
    shared_arrays = np.ndarray(buffer=buf, dtype='d', shape = (array_count,N,N,N))

    #Fence to synchronise processes after quering the window
    win_dict[color].Fence()

    return win_dict[color], shared_arrays


      ############################################################
def sum_beam(comm, data):
    """
    Sums data along a specified direction (X_BEAM or Y_BEAM) or a specific
    planes (XYPLANE or YZPLANE) using MPI and stores the value on the root 
    of that specific communicator. 
    
    Parameters:
    - comm: MPI communicator
    - data: Array of data to be summed
    """
    # Perform reduction (sum) along YBEAM (i.e., sum rows)
    reduced_data = np.zeros_like(data)
    comm.Reduce(data, reduced_data, op=MPI.SUM, root=0)

    return reduced_data

#################################################################################################################################

class Synchrotron:
    N: float = None             # number of grid points
    p: float = None             # electron number density exponent
    freq: float = None          # frequency of the radio waves
    c = 3e8                     # speed of light in m/s
    kpc = 1e3                   # scaled the dl in RM to kpc
    unit_magnetic = 15.7        # in micro Gauss
    n_e = 1e-3                  # number density in cm^{-3}
    L_0 = None                  # 
    pmax = None                 # maximum degree of polarisation
    Faraday_scaling = None
    Jy = 1e-23                  # Janskey in cgs unit
        
    @classmethod
    def set_class_variables(cls, N, p, freq):
        cls.N = N
        cls.p = p
        cls.freq = freq
        cls.L_0 = cls.N / (2 * np.pi)
        cls.pmax = (cls.p + 1) / (cls.p + 7.0 / 3)
        cls.Faraday_scaling = cls.n_e * cls.unit_magnetic * cls.L_0 * cls.kpc
    
    def __init__(self, los, rho, Bx, By, Bz, dx):
        self.los = los
        self.rho = rho
        self.Bx  = Bx
        self.By  = By
        self.Bz  = Bz
        self.dx  = dx
    
    def common(self):
        if self.los == 'z':
            B_parallel = self.Bz
            B_perp1    = self.Bx
            B_perp2    = self.By
            
        elif self.los == 'y':
            B_parallel = self.By
            B_perp1    = self.Bx
            B_perp2    = self.Bz
            
        elif self.los == 'x':
            B_parallel = self.Bx
            B_perp1    = self.By
            B_perp2    = self.Bz

        else:
            raise ValueError("Invalid line of sight. Use 'x', 'y', or 'z'.")

        # constants that needs to be multiplied for Jw is multiplied in next function
        # for memory efficiency
        Jw = (B_perp1**2 + B_perp2**2) ** ((self.p + 1) / 4)
        FD = 0.812 * self.Faraday_scaling * self.rho * B_parallel
        chi = np.pi / 2 + np.arctan2(B_perp2, B_perp1)

        del B_parallel
        del B_perp1
        del B_perp2
        
        return Jw, FD, chi

    def Faraday_depth(self, xaxis, yaxis, zaxis, FD):

        l = FD.shape[1]

        FD_los = np.zeros((l,l), dtype = 'd')

        if self.los == 'z':
            FD_los = np.trapz(FD, x=zaxis, axis=2)

        elif self.los == 'y':
            FD_los = np.trapz(FD, x=yaxis, axis=1)

        elif self.los == 'x':
            FD_los = np.trapz(FD, x=xaxis, axis=0)

        return FD_los

    def Stokes_parameters(self, xaxis, yaxis, zaxis, Jw, FD, chi):


        l = FD.shape[1]

        """
        By default, NumPy arrays use a C-like memory layout, where the last indices (typically rows)
	are stored in contiguous memory locations. This means that accessing these last indices first 
        allows them to be efficiently loaded into the cache. Therefore, it’s good practice to iterate 
	over the outer loops using the slower-changing indices, while accessing the rows (or last indices) 
        in the inner loops for better cache performance. To check whether a NumPy array (say FD) has a 
	C-contiguous memory layout, you can use the command FD.flags['C_CONTIGUOUS'], which returns True 
        if the array follows the C-style layout."""

	    
        FD_ds = np.zeros(l, dtype = 'd')
        chi_ds = np.zeros(l, dtype = 'd')
        Jomega = np.zeros(l, dtype = 'd')
        Q = np.zeros((l,l), dtype = 'd')
        U = np.zeros((l,l), dtype = 'd')
        I = np.zeros((l,l), dtype = 'd')
        
        omega = 2 * np.pi * self.freq
        prefactor = (self.unit_magnetic*1e-6)**2/self.Jy * omega ** (-(self.p - 1) / 2)
        freq_const = (Synchrotron.c**2 / self.freq**2)
        
        
        if self.los == 'z':

            for j in range(l):
                for k in range(l):
                    Jomega = prefactor * Jw[j,:,k]
                    FD_ds = np.trapz(FD[j, :, :k+1], x=zaxis[:k+1], axis=1) - 0.5 *FD[j, :, k] * self.dx
                    chi_ds = chi[j,:,k] + freq_const * FD_ds
                    if k==0 or k == l-1:
                        int_const = 0.5
                    else:
                        int_const = 1.0
                    Q[j,:] += int_const * Jomega * np.cos(2*chi_ds)
                    U[j,:] += int_const * Jomega * np.sin(2*chi_ds)
                    I[j,:] += int_const * Jomega
                #gc.collect()

        elif self.los == 'y':

            for j in range(l):
                for k in range(l):
                    Jomega = prefactor * Jw[j,k,:]
                    FD_ds = np.trapz(FD[j, :k+1, :], x=yaxis[:k+1], axis=0) - 0.5 *FD[j, k, :] * self.dx
                    chi_ds = chi[j,k,:] + freq_const * FD_ds
                    if k==0 or k == l-1:
                        int_const = 0.5
                    else:
                        int_const = 1.0
                    Q[j,:] += int_const * Jomega * np.cos(2*chi_ds)
                    U[j,:] += int_const * Jomega * np.sin(2*chi_ds)
                    I[j,:] += int_const * Jomega
                #gc.collect()
                    
        elif self.los == 'x':

            
            for j in range(l):
                for k in range(l):
                    Jomega = prefactor * Jw[k,j,:]
                    FD_ds  = np.trapz(FD[:k+1, j, :], x=xaxis[:k+1], axis=0) - 0.5 *FD[k, j, :] * self.dx
                    chi_ds = chi[k,j,:] + freq_const * FD_ds
                    if k==0 or k == l-1:
                        int_const = 0.5
                    else:
                        int_const = 1.0
                    Q[j,:] += int_const * Jomega * np.cos(2*chi_ds)
                    U[j,:] += int_const * Jomega * np.sin(2*chi_ds)
                    I[j,:] += int_const * Jomega
                #gc.collect()
                
        Q = 2 * np.pi * self.pmax * Q/l
        U = 2 * np.pi * self.pmax * U/l
        I = 2 * np.pi * I/l
                        
        del Jomega
        del chi_ds
        del FD_ds
        gc.collect()
        
        return Q, U, I


##############################################################################################################################
