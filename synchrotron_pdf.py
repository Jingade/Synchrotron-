import os
import gc
import h5py
import pencil as pc
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
from scipy.fftpack import fft, ifft, fft2
from scipy.fftpack import fftfreq
from mpi4py import MPI
from scipy import stats
from synchrotron import Synchrotron 
import synchrotron as syn

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# this creates a seperate communicator for each node
shared_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
node_size   = shared_comm.Get_size()

###########################################################################################################
# Data for grid creation 
start    = 12              # start of the var file
end      = 21             # end of the var file
nprocx   = 3              # x,y and z direction
nprocy   = end-start      # number of var files
nprocz   = 3              # number of frequencies

grid_dims = (nprocx, nprocy, nprocz)

# creates the grid of processors
ipx, ipy, ipz, XBEAM_COMM, YBEAM_COMM, ZBEAM_COMM, _, XZPLANE_COMM, _ = syn.create_grid_beam(*grid_dims)

#########################################################################################################3
# check for the layout of processors and raise error
node_name         = MPI.Get_processor_name()
unique_node_names = comm.allgather(node_name)
unique_node_names = sorted(set(unique_node_names))
node_rank         = unique_node_names.index(node_name)
num_nodes         = len(unique_node_names)

num_ipy_per_node  = nprocy//num_nodes
extra_ipy         = nprocy % num_nodes

if extra_ipy != 0:
    raise ValueError("number of ipy processors should be divisble by number of node.")

if num_ipy_per_node * nprocx * nprocz != node_size:
    raise ValueError("All nprocx*nprocz should lie within a node for a given ipy processors.")

#########################################################################################################

# rank 0 processors for different communicators
lfirst_procx  = (ipx==0)                    # for XBEAM_COMM
lfirst_procy  = (ipy==0)                    # for YBEAM_COMM
lfirst_procz  = (ipz==0)                    # for ZBEAM_COMM
lfirst_procxz = (ipx == 0 and ipz ==0)      # for XZPLANE_COMM

# Assigning the ipx and ipz processors
freq_list      = (0.5e9, 1e9, 6e9)          # Frequency in Hz (adjust as needed)
frequency      = freq_list[ipz]             # make sure nprocz is same as the number of frequencies
int_directions = ['z', 'y', 'x']
direction      = int_directions[ipx]        # make sure number of nprocx processor is same as number of direction


# Parameters
dim        = pc.read.dim()
N          = dim.nxgrid
bin_size   = 450
array_ipy  = 4                              # number of arrays to be shared
array_ipxy = 3
index      = ipx + nprocx * ipy

# parameters for data writing
postfix = ('_po5G','_1G', '_6G')
direct = ('Z', 'Y', 'X')
freq_append = postfix[ipz]
direct_append = direct[ipx]

# Define the folder name
data_dir = 'pdf_data'

# Create the folder if it doesn't exist
if rank == 0:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

"""
 creating the shared memory window corresponding to each ipy processors on each node. This mitigates 
 the bcast which unnecessarily creates the copies of Bx, By, Bz and rho on the same node. Additionally, 
 a window is created for Jomega, FD, chi computed from Bx, By, Bz and rho on ipz=0 processors, which 
 remains same for all the frequencies. This startegy avoids unnecessary copies of the data, making
 it memory efficient and optimises the RAM usage on each node. 
"""

# window is created for the data Bx, By, Bz and rho to be shared across XZPLANE processors
win_ipy, shared_arrays = syn.create_mem_window(lfirst_procxz, array_ipy, N, ipy, XZPLANE_COMM)

# creating the window for Jomega, FD, chi to be shared across ZBEAM processors 
win_common, shared_common = syn.create_mem_window(lfirst_procz, array_ipxy, N, index, ZBEAM_COMM)

##############################################################################################################

# load the data only along ipy processors 
if lfirst_procxz :
    i = start+ipy
    # 3D magnetic field distribution
    varf = 'VAR'+str(i)
    # change here to load your data files
    var = pc.read.var(var_file=varf, magic=['bb'], trimall=True)

    shared_arrays[0] = var.bb[0]
    shared_arrays[1] = var.bb[1]
    shared_arrays[2] = var.bb[2]
    shared_arrays[3] = np.exp(var.lnrho)
    
    xx = var.x
    yy = var.y
    zz = var.z
    dx = var.dx

    del var
    gc.collect()

    data = (xx, yy, zz, dx)
    XZPLANE_COMM.bcast(data, root=0)

else:

    data = XZPLANE_COMM.bcast(None, root=0)
    xx, yy, zz, dx = data

# Bx, By, Bz and rho points to the shared_arrays
Bx  = shared_arrays[0]
By  = shared_arrays[1]
Bz  = shared_arrays[2]
rho = shared_arrays[3]    


# since there is no cross talk between nodes. Synchronising the processes on each node is sufficient. 
shared_comm.Barrier()

##################################################################################################################

# assign the frequencies and initial parameters over the ipz processors
Synchrotron.set_class_variables(N=dim.nxgrid, p=3, freq=frequency)

# Initialize Synchrotron with the correct direction along ipx processors
# and instance synchrotron of the class 'Synchrotron' is created on all processors
synchrotron = Synchrotron(los=direction, rho=rho, Bx=Bx, By=By, Bz=Bz, dx=dx)


# calculate intermediate arrays only on the ipz = 0 processors
if lfirst_procz :
    shared_common[0], shared_common[1], shared_common[2] = synchrotron.common()

# point the Jomega, FD and chi to the shared_common, which is shared across other ipz processors
Jw  = shared_common[0]
FD  = shared_common[1]
chi = shared_common[2]


# Freeing Bx, By, Bz and rho, which is no longer needed now
win_ipy.Free()
shared_comm.Barrier()

# Computing the Faraday depth along los. There is no frequnecy dependencies here. 
if lfirst_procz :
    FD_los = synchrotron.Faraday_depth(xx, yy, zz, FD)
    FD_flat, binFD = np.histogram(FD_los.ravel(), bins=bin_size, density=True)
    data = np.column_stack((binFD[:-1], FD_flat))
    reduced_FD = syn.sum_beam(YBEAM_COMM, data)
    if ipy == 0:
        reduced_FD /= (end-start)
        fileFD = data_dir+'/pdf_FD_'+direct_append+'.txt'
        np.savetxt(fileFD, reduced_FD, delimiter=' ', header='FD---pdfFD', comments='',fmt='%.4e')
    
# Perform the computation over all the processors to get Stokes parameters
Q, U, I = synchrotron.Stokes_parameters(xx, yy, zz, Jw, FD, chi)

# free all the memory window, which are not necessary
win_common.Free()
comm.Barrier()
########################################################################################################################

# compute the polarised intensity and normalised Q, U over all processors
PI = np.hypot(Q, U)
FP = np.empty_like(PI)

np.divide(Q, I, out=Q)
np.divide(U, I, out=U)
np.divide(PI, I, out=FP)

#compute the integral scale of the polarised emission parameters
PI_fft = fft2(PI)
FP_fft = fft2(FP)

spec_PI = np.abs(PI_fft)**2
spec_FP = np.abs(FP_fft)**2

#Int_scale calculation                                                                                                                                                                                     
int_scalePI = np.trapz(spec_PI[0,:], dx = 1)
normPI = np.sum(spec_PI)
PI_scale_x = int_scalePI/normPI
int_scalePI = np.trapz(spec_PI[:,0], dx=1)
PI_scale_y = int_scalePI/normPI

int_scaleFP = np.trapz(spec_FP[0,:], dx = 1)
normFP = np.sum(spec_FP)
FP_scale_x = int_scaleFP/normFP
int_scaleFP = np.trapz(spec_FP[:,0], dx=1)
FP_scale_y = int_scaleFP/normFP


data_PI = np.array([PI_scale_x, PI_scale_y, FP_scale_x, FP_scale_y])
reduce_dataPI = syn.sum_beam(YBEAM_COMM, data_PI)        # summing the data along varfiles                                                                                                                 

if lfirst_procy:
    reduce_dataPI /= (end-start)
    int_data = [direct_append, freq_append, *reduce_dataPI]
    all_data = XZPLANE_COMM.gather(int_data, root = 0)

    if lfirst_procxz:
        file_name = data_dir+'/intscale_PI_FP.txt'
        with open(file_name, "w") as f:
        # Define format: two strings followed by four floats with six decimal places                                                                                                                       
            line_format = "{:s} {:s} {:.4f} {:.4f} {:.4f} {:.4f}\n"
            f.write("direct---freq---PI_x---PI_y---FP_x---FP_y\n")
            # Write each data entry using the defined format                                                                                                                                               
            for data in all_data:
                string_part = data[:2]
                float_part = [float(x) for x in data[2:]]
                f.write(line_format.format(*string_part, *float_part))


# compute the histogram for normalised stokes parameters and synchrotron emissivity
FP_flat, binFP = np.histogram(FP.ravel(), bins=bin_size, density=True)
Q_flat,  binQ  = np.histogram(Q.ravel(),  bins=bin_size, density=True)
U_flat,  binU  = np.histogram(U.ravel(),  bins=bin_size, density=True)
PI_flat, binPI = np.histogram(PI.ravel(), bins=bin_size, density=True)
I_flat,  binI  = np.histogram(I.ravel(),  bins=bin_size, density=True)

# pack the PDFs into single array 
data1 = np.column_stack((binFP[:-1], FP_flat, binQ[:-1], Q_flat, binU[:-1], U_flat))
data2 = np.column_stack((binI[:-1], I_flat, binPI[:-1], PI_flat))

#average the PDFs over the varfiles, that is along the ipy processors and
# result is available on ipy = 0 processors
reduce_data1 = syn.sum_beam(YBEAM_COMM, data1)        # summing the data along varfiles 
reduce_data2 = syn.sum_beam(YBEAM_COMM, data2)        # which is loaded in ipy processors


"""
 Calculating the normalisation constant for 1GHz frequency which
 is lying in the ipz = 1 processors
"""

if ipz == 1:
    Isum = np.sum(I)*dx*dx
    reduce_Isum = syn.sum_beam(YBEAM_COMM, Isum)
    reduce_Isum /= (end-start)   #reduce_Isum lies on ipy=0 processors

    # share the reduce_Isum with other processors
    if ipx == 0:
        XBEAM_COMM.bcast(reduce_Isum, root=0)
    else:
        reduce_Isum = XBEAM_COMM.bcast(None, root=0)

    #share the reduce_Isum with other ipz processors
    ZBEAM_COMM.bcast(reduce_Isum, root=1)
else:
    reduce_Isum = ZBEAM_COMM.bcast(None, root=1)
    
#########################################################################################################################

# writing the data to the files on ipy=0 and ipy=5 processors

if ipy == 5:
    map_data = data_dir+'/avg_map'+direct_append+freq_append+'.h5'
    with h5py.File(map_data, 'w') as f:
        f.create_dataset('Intensity', data=I)
        f.create_dataset('Polarised_Intensity', data=PI)
        f.create_dataset('Fractional_Polarisation', data=FP)
        if direction == 'z':
            f['Intensity'].attrs['axis_0'] = 'X'
            f['Intensity'].attrs['axis_1'] = 'Y'
            f['Polarised_Intensity'].attrs['axis_0'] = 'X'
            f['Polarised_Intensity'].attrs['axis_1'] = 'Y'
            f['Fractional_Polarisation'].attrs['axis_0'] = 'X'
            f['Fractional_Polarisation'].attrs['axis_1'] = 'Y'
        elif direction == 'y':
            f['Intensity'].attrs['axis_0'] = 'Z'
            f['Intensity'].attrs['axis_1'] = 'X'
            f['Polarised_Intensity'].attrs['axis_0'] = 'Z'
            f['Polarised_Intensity'].attrs['axis_1'] = 'X'
            f['Fractional_Polarisation'].attrs['axis_0'] = 'Z'
            f['Fractional_Polarisation'].attrs['axis_1'] = 'X'
        elif direction == 'x':
            f['Intensity'].attrs['axis_0'] = 'Y'
            f['Intensity'].attrs['axis_1'] = 'Z'
            f['Polarised_Intensity'].attrs['axis_0'] = 'Y'
            f['Polarised_Intensity'].attrs['axis_1'] = 'Z'
            f['Fractional_Polarisation'].attrs['axis_0'] = 'Y'
            f['Fractional_Polarisation'].attrs['axis_1'] = 'Z'
            

if ipy ==0:
    reduce_data1 /= (end-start)
    reduce_data2 /= (end-start)
    XZPLANE_COMM.Barrier()
    print(direction , 'Isum=', reduce_Isum)
    fileSync = data_dir +'/pdf_Sync'+direct_append+freq_append+'.txt'
    fileInt = data_dir+'/pdf_int'+direct_append+freq_append+'.txt'
    np.savetxt(fileSync, reduce_data1, delimiter=' ', header='FP---pdfFP---Q----pdfQ---U---pdfU', comments='',fmt='%.4e')
    np.savetxt(fileInt, reduce_data2, delimiter=' ', header=f'I---pdfI---PI---pdfPI--{reduce_Isum:.2f}', comments='',fmt='%.4e')

    
