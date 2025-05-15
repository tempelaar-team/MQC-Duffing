import numpy as np
# using atomic unit
kT=0.0
hbar=1
c0=1.0                  #speed of light in au
epsilon=1.0         #vacuum permittivity in atomic unit

# Cavity setup
#l=50.0                   # cavity length
l=np.pi/50
#alpha=np.arange(1,400+1,2)
alpha=np.array([1])
N=alpha.size
#w=np.array([50.0]) #(np.pi * c0 * alpha)/l    # cavity mode frequency
w=(np.pi * c0 * alpha)/l    # cavity mode frequency

# Atom setup
#mu=np.array([1.0])
mu=np.array([np.sqrt(np.pi)/50])
num_atom=1
energy=np.array([0.0, 50])
r_atom = np.array([l/2])

# Running time setup
tmax=10
dt=0.001
savestep=10

calcdir = 'data'

# for plotting
r_resolution=600
intens_save_t=100