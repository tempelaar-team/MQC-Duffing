import numpy as np
import time
import scipy.sparse
from numba import jit #, prange
import os
from shutil import copyfile
# Import the parameters
from input import *

start_time_all = time.time()

#Hamiltonian
start_time_H=time.time()

#assign index
#N=int(N)
w = np.append(0, w)
alpha = np.append(0, alpha)

#assign index for modes
NumDiff2modes=int((N)*(N-1)/2)
num_modesbasis=int(2*N+1+NumDiff2modes)
index_vac = 0
index_onephoton = 1 + index_vac
index_twophoton = int(N + index_onephoton)
index_threephoton = num_modesbasis
indexab=np.zeros((num_modesbasis, 2))

iter= int(1+2*N)
indexab[0]=((0,0))
for i in range(1,int(N+1)):
    indexab[i] = ((i, 0))
    indexab[int(i+N)] = ((i, i))
    for j in range(i + 1, int(N+1)):
        indexab[iter] = ((i,j))
        iter = iter+1

#assign index with atoms
total_number=int(energy.size**num_atom*num_modesbasis)
index = np.zeros((total_number, int(num_atom+2)))

for n in range(num_atom):
    iii = 0
    for i in range(energy.size**(num_atom-n)):
        index[int(iii):int(iii+(n+1)*(num_modesbasis)), int(num_atom-1-n)] = i%energy.size
        if n == 0:
            index[int(iii):int(iii + num_modesbasis), int(num_atom):int(num_atom+2)] =indexab
        iii += (n+1)*(num_modesbasis)

# make index for indexab_d and indexabc_d
iter=0
indexab_d = np.zeros((int((N)*(N-1)/2),2), dtype=int)
for i in np.arange(1, int(N+1)):
    for j in np.arange(i+1, int(N+1)):
        indexab_d[iter] = ((i,j))
        iter +=1

#@jit(nopython=True, fastmath=True)#, parallel=True)
def get_H():
    #Hindex=np.array([[0, 0]], dtype='int64')
    H=np.array([energy[0]], dtype='float64')
    ni=nj=0
    #generate the diagonal terms
    Hii = [energy[int(index[i][ni])] + hbar * w[int(index[i][1])]
         + hbar * w[int(index[i][2])] for i in range(1, total_number)]
    Hi = [i for i in range(0, total_number)]
    Hj = Hi.copy()
    H = np.concatenate((H, np.array(Hii)))

    # Only focus on the offdiagonal terms from the middle electronic state
    # zero photon state
    # |e, 0, 0, 0> -> |g, i, 0, 0>
    nowi = index_vac + num_modesbasis
    Hij = np.sqrt((w[1:]) / (epsilon * l)) * np.sin(np.pi * alpha[1:] * r_atom[nj] / l)
    H = np.concatenate((H, mu[0] * Hij))
    nowHi= [nowi for j in range(index_onephoton, index_twophoton)]
    Hi += nowHi
    Hj += [j for j in range(index_onephoton, index_twophoton)]

    # one photon state
    # |e, i, 0, 0> -> |g, i, i, 0>
    H = np.concatenate((H, np.sqrt(2)*mu[0] * Hij, mu[0] * Hij))
    nowHi = [i for i in range(num_modesbasis + index_onephoton, num_modesbasis + index_twophoton)]
    Hi += nowHi
    Hj += [j for j in range(index_twophoton, int(index_twophoton + N))]
    # |e, i, 0, 0> -> |g, 0, 0, 0> #CRW
    Hi += nowHi
    Hj += [index_vac for j in range(num_modesbasis + index_onephoton, num_modesbasis + index_twophoton)]
    # |e, i (j), 0, 0> <- |g, i, j, 0>
    index_for_ab=0
    nowHij=[]
    for i in range(int(index_twophoton + N), index_threephoton):
        Hi += [i, i]
        nowHj =(num_modesbasis + index_onephoton + indexab_d[index_for_ab] - 1).tolist()
        Hj += nowHj
        preHij = np.sqrt((w[indexab_d[index_for_ab]]) / (epsilon * l)) * np.sin(
                np.pi * alpha[indexab_d[index_for_ab]] * r_atom[nj] / l)
        nowHij += (mu[0]*np.flip(preHij)).tolist()

        index_for_ab+=1
    H = np.concatenate((H, nowHij))

    # two photon state
    # |e, i, i, 0> -> |g, i, 0, 0> #CRW
    H = np.concatenate((H, np.sqrt(2)*mu[0] * Hij))
    nowHi = [i for i in range(int(num_modesbasis + index_twophoton), int(num_modesbasis + index_twophoton+N))]
    Hi += nowHi
    Hj += [j for j in range(index_onephoton, index_twophoton)]

    index_for_ab=0
    nowHij=[]
    for i in range(int(num_modesbasis+index_twophoton + N), num_modesbasis+index_threephoton):
        # |e, i, j, 0> -> |g, i (j), 0, 0> #CRW
        Hi += [i, i]
        nowHj = (index_onephoton + indexab_d[index_for_ab] - 1).tolist()
        Hj += nowHj
        preHij = np.sqrt((w[indexab_d[index_for_ab]]) / (epsilon * l)) * np.sin(
                np.pi * alpha[indexab_d[index_for_ab]] * r_atom[nj] / l)
        nowHij += (mu[0]*np.flip(preHij)).tolist()

        index_for_ab+=1
    H = np.concatenate((H, nowHij))
    Hindex = np.vstack((np.array(Hi), np.array(Hj))).transpose()
    return Hindex, H

@jit(nopython=True, fastmath=True)#, parallel=True)
def my_matmul(Hindex, H, wavefn):
    wavefn_new=np.zeros_like(wavefn, dtype='complex')
    for h in range(H.size):
        i = int(Hindex[h,0])
        j = int(Hindex[h,1])
        Hij = H[h]
        if i == j:
            wavefn_new[i] += Hij * wavefn[i]
        else:
            wavefn_new[j] += Hij * wavefn[i]
            wavefn_new[i] += Hij * wavefn[j]
    return wavefn_new

@jit(nopython=True, fastmath=True)
def RK4(Hindex, H, wavefn, dt):
    K1 = -1j / hbar * my_matmul(Hindex, H, wavefn)
    K2 = -1j / hbar * my_matmul(Hindex, H, (wavefn + 0.5 * dt * K1))
    K3 = -1j / hbar * my_matmul(Hindex, H, (wavefn + 0.5 * dt * K2))
    K4 = -1j / hbar * my_matmul(Hindex, H, (wavefn + dt * K3))
    wavefn_new = wavefn + dt * 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    return wavefn_new

def get_eig(H):
    sparse_H=scipy.sparse.csc_matrix(H)
    evals, evecs = scipy.sparse.linalg.eigsh(sparse_H, k=1, which='SA')
    return evals, evecs

@jit(nopython=True, fastmath=True)#, parallel=True)
def time_prop(all_ini_wavefn, Hindex, H):
    wavefn = all_ini_wavefn
    maxstep = int(tmax/dt +1)
    wavefn_save = np.zeros((int(maxstep/savestep+1), all_ini_wavefn.size), dtype='complex128')
    wavefn_save[0] = wavefn

    for t in range(1, maxstep):
        wavefn = RK4(Hindex, H, wavefn, dt)
        if t % savestep == 0:
            wavefn_save[int(t/savestep)] = wavefn
        if t == maxstep-1:
            wavefnend = wavefn
    return wavefn_save, wavefnend

Hindex, H = get_H()

all_ini_wavefn=np.zeros(total_number, dtype='complex128')

end_time_H=time.time()
print('Hamiltonian generated. Basis size = '+ str(total_number) +', half of H non-zero size = '+ str(H.size) + ', N=' + str(N))
print('Sparsity = ' + str(H.size*2/total_number**2))
print('Running Wall Time = %10.3f second' % (end_time_H - start_time_H))

start_time_eigh=time.time()
Hsparse = scipy.sparse.coo_matrix((np.concatenate((H, H[total_number:])),
                             (np.concatenate((Hindex[:,0],Hindex[total_number:,1])),
                              np.concatenate((Hindex[:,1],Hindex[total_number:,0])))))
Hsparse = Hsparse.tocsr()
evals, evecs = get_eig(Hsparse)
end_time_eigh=time.time()
#
print('Found the lowest eigenstate. Running Wall Time = %10.3f second' % (end_time_eigh - start_time_eigh))

start_time_run=time.time()
all_ini_wavefn[np.where((index[:,0] == 1) & (index[:,1] == 0) & (index[:,2] == 0))]=1 # |e, 0, 0>
#all_ini_wavefn=evecs[:,0] # "Real ground state"

all_ini_wavefn=all_ini_wavefn/np.linalg.norm(all_ini_wavefn)

wavefn_save, wavefnend=time_prop(all_ini_wavefn.astype(complex), Hindex, H)

end_time_all = time.time()

if not (os.path.exists(calcdir)):
    os.mkdir(calcdir)
copyfile('input.py', calcdir + '/inputfile_bk.txt')

np.save(calcdir + '/index.npy', index)
np.savez_compressed(calcdir + '/rho.npz', rho=wavefn_save)
np.savetxt(calcdir + '/final_rho.csv', wavefnend, delimiter=',')

print('Finished time evolution. Running Wall Time = %10.3f second' % (end_time_all - start_time_run))
print('Calculation Finished. Running Wall Time = %10.3f second' % (end_time_all - start_time_all))
