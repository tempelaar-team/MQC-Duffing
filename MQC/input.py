import numpy as np
# using atomic unit
kT=0.0
hbar=1
c0=1.0                  # speed of light in au
epsilon=1.0         # vacuum permittivity in atomic unit

# Cavity setup
l=np.pi/50                  # cavity length
#l=1092.67*2*0.5             # smallest
#alpha=np.arange(1,400+1,2)  # cavity mode
alpha=np.array([1])
N = alpha.size
w=(np.pi * c0 * alpha)/l    # cavity mode frequency

# Atom setup
num_atom=1
energy=np.array([0.0, 50])
mu12=np.sqrt(np.pi)/50
ini_wavefn=np.array([1, 0])
r_atom = np.array([l/2])

# running time setup
tmax=10
dt=0.001
savestep=10

proc = 1                       # number of processors
num_trj = 1                      # for each processor. Recommend 10 to 100
total_trj = 1           # total number of trajectories. total_trj/(proc*num_trj) will define the number of runs and should be a integer.
calcdir = 'data'
status = 'NORESTART'  #'NORESTART' or 'RESTART'

r_resolution=100
intens_save_t=10