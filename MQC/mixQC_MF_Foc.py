import numpy as np
from numba import jit
from os import path
import ray
import time
# Import the parameters
from input import *

wgrid = w[:,np.newaxis]

if num_atom != np.array(ini_wavefn).size/2 or num_atom != np.array(r_atom).size:
    print('WARNING! the number of atoms does not equal to given wavefn or positions.')

def Wigner():
    # Wigner sampling
    # stdP = np.sqrt(hbar * w / (2))
    # stdQ = np.sqrt(hbar / (2 * w))
    # P0 = np.random.normal(0.0, stdP)
    # Q0 = np.random.normal(0.0, stdQ)

    # Focused sampling using n0=0.59
    theta = np.random.uniform(0, 2*np.pi, w.size)
    E = 0.59*hbar*w #0.5*hbar*w 
    theta = 0 #setting P=0, Q>0 
    Q0 = np.sqrt(2*E) * np.cos(theta)/w
    P0 = np.sqrt(2*E) * np.sin(theta)
    return Q0, P0

def Boltzmann():
    avg = 0
    stdP = np.ones_like(w)*np.sqrt(kT)
    stdQ = np.sqrt(kT/w**2)
    P0 = np.random.normal(avg, stdP)
    Q0 = np.random.normal(avg, stdQ)
    return np.zeros_like(Q0), np.zeros_like(P0)

#@jit(nopython=True)
def calc_QMdyn_get_QF(wavefn, Q, Qzp, lambda_al):
    QF = np.zeros_like(Q)
    QFzp = np.zeros_like(Q)
    wavefn_adb = np.zeros_like(wavefn)
    for n in range(num_atom):
        Hq = H_q(Q, Qzp, wavefn[2*n: 2*n+int(ini_wavefn.size+1)], lambda_al[:,n])
        now_wavefn, now_wavefn_adb = prop_Q_exact(Hq, wavefn[2*n: 2*n+int(ini_wavefn.size+1)], dt)
        wavefn[2*n] = now_wavefn[0]
        wavefn[2*n+1] = now_wavefn[1]
        wavefn_adb[2*n] = now_wavefn_adb[0]
        wavefn_adb[2*n+1] = now_wavefn_adb[1]
        nowQF = Q_F(now_wavefn, lambda_al[:, n])
        #QF += nowQF
        factor = 1
        QFzp += factor * nowQF
    return wavefn, wavefn_adb, QF, QFzp

#@jit(nopython=True)
def Q_F(wavefn, lambda_alpha):
    force = -2 * wgrid * mu12 * lambda_alpha[:,np.newaxis] * np.real(np.conjugate(wavefn[0]) * wavefn[1])
    return force

@jit(nopython=True)
def H_q(Q, Qzp, wavefn, lambda_alpha):
    Hq = np.zeros((ini_wavefn.size, ini_wavefn.size, num_trj))
    for i in range(ini_wavefn.size):
        Hq[i, i, :] = energy[i]
        for j in range(i+1, ini_wavefn.size):
            intQ = Qzp
            Hq[i, j, :] = np.sum(mu12 * wgrid * lambda_alpha[:,np.newaxis] * intQ, axis=0)
            Hq[j, i, :] = Hq[i, j, :]
    return Hq

@jit(nopython=True)
def prop_Q_exact(Hq, psiA, dt):
    psiA_db = psiA.copy()
    psiA_adb = np.zeros_like(psiA)
    for n in range(num_trj):
        evals, evecs =  np.linalg.eigh(Hq[:,:,n])
        evalsA_exp = np.exp(-1.0j * evals * dt)
        psiA_adb[:,n] = evecs.astype('complex128').T@np.ascontiguousarray(psiA[:,n])
        psiA_adb[:,n] = evalsA_exp * psiA_adb[:,n]
        psiA_db[:,n] = evecs.astype('complex128')@np.ascontiguousarray(psiA_adb[:,n])
    return psiA_db, psiA_adb

#@jit(nopython=True)
def quantum_energy(wavefn, Q, Qzp, lambda_alpha):
    bra = np.conjugate(wavefn)
    q_energy=np.zeros(num_trj, dtype=float)
    qc_energy=np.zeros(num_trj, dtype=float)
    for i in range(ini_wavefn.size):
        q_energy += np.abs(wavefn[i]) ** 2 * energy[i]
        for j in range(i+1, ini_wavefn.size):
            intQ = Qzp
            qc_energy += np.sum(mu12 * wgrid * lambda_alpha[:,np.newaxis] * intQ, axis=0) * 2 * np.real(wavefn[i] * bra[j])
    return q_energy, qc_energy

@jit(nopython=True)
def get_diff_eq(wgrid, P, Q, Pzp, Qzp, QF, QFzp):
    Qdot = P
    Pdot = -wgrid ** 2 * Q + QF
    Qdotzp = Pzp
    Pdotzp = -wgrid ** 2 * Qzp + QFzp
    return Qdot, Pdot, Qdotzp, Pdotzp

@jit(nopython=True)
def RK4_new(wgrid, P, Q, Pzp, Qzp, QF, QFzp, dt):
    K1, L1, K1zp, L1zp = get_diff_eq(wgrid, P, Q, Pzp, Qzp, QF, QFzp)
    K2, L2, K2zp, L2zp = get_diff_eq(wgrid, P + 0.5*dt*L1, Q + 0.5*dt*K1, Pzp + 0.5*dt*L1zp, Qzp + 0.5*dt*K1zp, QF, QFzp)
    K3, L3, K3zp, L3zp = get_diff_eq(wgrid, P + 0.5*dt*L2, Q + 0.5*dt*K2, Pzp + 0.5*dt*L2zp, Qzp + 0.5*dt*K2zp, QF, QFzp)
    K4, L4, K4zp, L4zp = get_diff_eq(wgrid, P + dt*L3, Q + dt*K3, Pzp + dt*L3zp, Qzp + dt*K3zp, QF, QFzp)
    new_P = P + 0.166667 * dt * (L1 + 2 * L2 + 2 * L3 + L4)
    new_Q = Q + 0.166667 * dt * (K1 + 2 * K2 + 2 * K3 + K4)
    new_Pzp = Pzp + 0.166667 * dt * (L1zp + 2*L2zp + 2*L3zp + L4zp)
    new_Qzp = Qzp + 0.166667 * dt * (K1zp + 2*K2zp + 2*K3zp + K4zp)
    return new_P, new_Q, new_Pzp, new_Qzp

#@jit(nopython=True)
def class_energy(Q, P):
    c_energy = 1/2 * np.sum((P**2) + (wgrid**2)*(Q**2), axis=0)
    return c_energy

#@jit(nopython=True)
def get_dipole(wavefn, muwavefn):
    mu0mut = mu12**2 * (np.conjugate(wavefn[0]) * muwavefn[1] + np.conjugate(wavefn[1]) * muwavefn[0])
    return mu0mut

#@jit(nopython=True)
def get_Nph(Q, P, Qzp, Pzp):
    Nph = 0.5 * (P**2 + wgrid**2 * Q**2)/(hbar*wgrid) 
    Nphzp = 0.5 * (Pzp**2 + wgrid**2 * Qzp**2)/(hbar*wgrid)
    return Nph, Nphzp

#@jit(nopython=True)
def get_E_E2(Q, Qzp, E2zp0=np.zeros((r_resolution, num_trj))):
    r = np.linspace(0, l, r_resolution)
    zeta = np.sqrt(hbar*wgrid/(epsilon*l)) * np.sin(alpha[:,np.newaxis]*np.pi/l * r)
    E = np.sum(np.sqrt(2*wgrid[:,:,np.newaxis]/hbar) * zeta[:,:,np.newaxis] * Q[:,np.newaxis,:], axis=0)
    E2 = E**2
    Ezp = np.sum(np.sqrt(2*wgrid[:,:,np.newaxis]/hbar) * zeta[:,:,np.newaxis] * Qzp[:,np.newaxis,:], axis=0)
    E2zp = Ezp**2 - E2zp0
    return E, E2, Ezp, E2zp

#@jit(nopython=True)
def get_E_correlate(E0, Ezp0, E, Ezp):
    E0Etall = (E0 + Ezp0) * (E + Ezp) 
    return E0Etall

#@jit(nopython=True)
def get_Q_correlate(Q0, Qzp0, Q, Qzp, Qzp_noQF):
    Q0Qtall = (Q0 + Qzp0) * (Q + Qzp) #- Qzp0 * Qzp_noQF
    Q0Qtall = Qzp
    return Q0Qtall

def run_dyn_propogation(ini_wavefn, Q, P, Qzp, Pzp, lambda_al):
    wavefn = ini_wavefn.copy()
    wavefn_adb = ini_wavefn.copy()
    muwavefn = np.zeros_like(wavefn)
    muwavefn[0] = 1.0
    E0, allE20, Ezp0, E2zp0 = get_E_E2(Q, Qzp)
    Q0 = Q.copy()
    Qzp0 = Qzp.copy()
    Q_noQF = Q.copy()
    P_noQF = P.copy()
    Qzp_noQF = Qzp.copy()
    Pzp_noQF = Pzp.copy()
    Qg = Q.copy()
    Pg = P.copy()
    Qzpg = Qzp.copy()
    Pzpg = Pzp.copy()
    maxstep = int(tmax/dt+1)
    wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
        dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save = create_array(maxstep)
    wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
            dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save = \
            save_things(0, wavefn, wavefn_adb, muwavefn, Q, P, Qzp, Pzp, Qzp_noQF, lambda_al, E0, Ezp0, E2zp0, Q0, Qzp0,
                sys_energy_save, wavefnsquare_save, wavefnsquare_save_adb,
                dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save)
    
    savepoint = 1
    for t in range(1, maxstep):
        wavefn, wavefn_adb, QF, QFzp = calc_QMdyn_get_QF(wavefn, Q, Qzp, lambda_al)
        #muwavefn, muwavefn_adb, QFg, QFzpg = calc_QMdyn_get_QF(muwavefn, Qg, Qzpg, lambda_al)
        P, Q, Pzp, Qzp = RK4_new(wgrid, P, Q, Pzp, Qzp, np.zeros_like(QF), QFzp, dt)
        #Pg, Qg, Pzpg, Qzpg = RK4_new(wgrid, Pg, Qg, Pzpg, Qzpg, np.zeros_like(QF), QFzpg, dt)

        # saving data
        if t % savestep == 0:
            wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
            dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save = \
            save_things(savepoint, wavefn, wavefn_adb, muwavefn, Q, P, Qzp, Pzp, Qzp_noQF, lambda_al, E0, Ezp0, E2zp0, Q0, Qzp0,
                sys_energy_save, wavefnsquare_save, wavefnsquare_save_adb,
                dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save)
            savepoint += 1
        # saving the end points
        if t == maxstep-1:
            Pend = P
            Pendzp = Pzp
            Qend = Q
            Qendzp = Qzp
            wavefnend = wavefn
    return wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
        dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save,\
            wavefnend, Pend, Qend, Pendzp, Qendzp

@ray.remote
def run_dyn(index, ini_wavefn, Q, P, Qzp, Pzp, lambda_al):
    wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
        dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save,\
            wavefnend, Pend, Qend, Pendzp, Qendzp = run_dyn_propogation(ini_wavefn, Q, P, Qzp, Pzp, lambda_al)
    return wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
        dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save,\
            wavefnend, Pend, Qend, Pendzp, Qendzp

def create_array(maxstep):
    totalsavepoints=int(maxstep/savestep+1)
    wavefnsquare_save = np.zeros((totalsavepoints, num_atom*energy.size))
    wavefnsquare_save_adb = np.zeros((totalsavepoints, num_atom*energy.size))
    sys_energy_save = np.zeros((totalsavepoints, num_atom*2+2))
    dipole_save = np.zeros((totalsavepoints, num_atom), dtype='complex128')
    Nph_save = np.zeros((totalsavepoints, N))
    Nphzp_save = np.zeros((totalsavepoints, N))
    E2_save = np.zeros((totalsavepoints, r_resolution))
    E2zp_save = np.zeros((totalsavepoints, r_resolution))
    E0Etall_save = np.zeros((totalsavepoints, r_resolution))
    Q0Qtall_save = np.zeros((totalsavepoints, N))
    return wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
        dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save

def save_things(t, wavefn, wavefn_adb, muwavefn, Q, P, Qzp, Pzp, Qzp_noQF, lambda_al, E0, Ezp0, E2zp0, Q0, Qzp0,
                sys_energy_save, wavefnsquare_save, wavefnsquare_save_adb,
                dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save):
    wavefnsquare_save[t] = np.sum(np.real(wavefn * np.conjugate(wavefn)), axis=1)
    wavefnsquare_save_adb[t] = np.sum(np.real(wavefn_adb * np.conjugate(wavefn_adb)), axis=1)
    qE = np.zeros((num_atom, num_trj))
    qcE = np.zeros((num_atom, num_trj))
    for n in range(num_atom):
        qE[n, :], qcE[n, :] = quantum_energy(wavefn[2*n: 2*n+int(ini_wavefn.size+1)], Q, Qzp, lambda_al[:, n])
    sys_energy_save[t,0] = np.sum(class_energy(Q, P), axis=0)
    sys_energy_save[t,1] = np.sum(class_energy(Qzp, Pzp), axis=0)
    sys_energy_save[t,2:2+num_atom] = np.sum(qE, axis=1)
    sys_energy_save[t,2+num_atom:2+2*num_atom] = np.sum(qcE, axis=1)
    dipole_save[t] = np.sum(get_dipole(wavefn, muwavefn), axis=0)
    allNph, allNphzp = get_Nph(Q, P, Qzp, Pzp)
    Nph_save[t] = np.sum(allNph, axis=1)
    Nphzp_save[t] = np.sum(allNphzp, axis=1)
    allE, allE2, allEzp, allE2zp = get_E_E2(Q, Qzp, E2zp0)
    E2_save[t] = np.sum(allEzp, axis=1) #TODO: this is changed to allEzp
    E2zp_save[t] = np.sum(allE2zp, axis=1)
    E0Etall_save[t] = np.sum(get_E_correlate(E0, Ezp0, allE, allEzp), axis=1)
    Q0Qtall_save[t] = np.sum(get_Q_correlate(Q0, Qzp0, Q, Qzp, Qzp_noQF), axis=1)
    return wavefnsquare_save, wavefnsquare_save_adb, sys_energy_save, \
        dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save

def initial_lambda():
    now_r = r_atom.copy()
    lambda_al = np.zeros((int(N), num_atom))
    for n in range(num_atom):
        lambda_al[:, n] = np.sqrt(2/(epsilon*l)) * np.sin(np.pi * alpha * now_r[n] / l)
    return lambda_al

def initial_wavefn(ini_wavefn, num_trj, proc):
    wavefn = np.zeros((2 * num_atom, num_trj), dtype='complex128')
    for n in range(num_atom):
        wavefn[n * 2, :] = ini_wavefn[n * 2]
        wavefn[n * 2 + 1, :] = ini_wavefn[n * 2 + 1]
    wavefnsplit=np.array(np.array_split(wavefn, proc, axis=1))
    return wavefnsplit

#@jit(nopython=True)
def init_classical_parallel(num_trj, proc):
    Q = np.zeros((int(N), num_trj))
    P = np.zeros((int(N), num_trj))
    Qzp = np.zeros((int(N), num_trj))
    Pzp = np.zeros((int(N), num_trj))
    for n in range(num_trj):
        Q[:, n], P[:, n] = Boltzmann()
        Qzp[:, n], Pzp[:, n] = Wigner()
    Qsplit=np.array(np.array_split(Q, proc, axis=1))
    Psplit=np.array(np.array_split(P, proc, axis=1))
    Qsplitzp=np.array(np.array_split(Qzp, proc, axis=1))
    Psplitzp=np.array(np.array_split(Pzp, proc, axis=1))
    return Qsplit, Psplit, Qsplitzp, Psplitzp

def parallel_run_ray(total_trj, proc):
    trials = proc# int(total_trj/num_trj)
    r_ind = 0
    lambda_al = initial_lambda()
    ray.init()
    for run in range(int(total_trj/(proc*num_trj))):
        print('Executing run number '+str(run))
        if 'status' in globals() and status=='RESTART':
            global calcdir
            old_calcdir = calcdir
            calcdir = calcdir + '/RESTART'
            print('new calculation directory: ' + str(calcdir))
            wavefnall = np.load(old_calcdir + '/wavefnend.npz')['wavefnend']
            qall = np.load(old_calcdir + '/Qend.npz')['Qend']
            qallzp = np.load(old_calcdir + '/Qendzp.npz')['Qendzp']
            pall = np.load(old_calcdir + '/Pend.npz')['Pend']
            pallzp = np.load(old_calcdir + '/Pendzp.npz')['Pendzp']
            wavefn = np.array(np.array_split(wavefnall, proc, axis=-1))
            q = np.array(np.array_split(qall, proc, axis=-1))
            Qzp = np.array(np.array_split(qallzp, proc, axis=-1))
            p = np.array(np.array_split(pall, proc, axis=-1))
            Pzp = np.array(np.array_split(pallzp, proc, axis=-1))
        else:
            wavefn = initial_wavefn(ini_wavefn, proc * num_trj, trials)
            q, p, Qzp, Pzp = init_classical_parallel(proc * num_trj, trials)

        results = [run_dyn.remote(i, wavefn[i], q[i], p[i], Qzp[i], Pzp[i], lambda_al) for i in range(proc)]
        for r in results:
            (simRhoA, simRhoA_adb, simE, dipole_save, Nph_save, Nphzp_save, E2_save, E2zp_save, E0Etall_save, Q0Qtall_save,\
            wavefnend, Pend, Qend, Pendzp, Qendzp) = ray.get(r)
            if run == 0 and r_ind == 0:
                simEdat = np.zeros_like(simE)
                simRhoAdat = np.zeros_like(simRhoA)
                simRhoAdat_adb = np.zeros_like(simRhoA_adb)
                dipole_save_dat = np.zeros_like(dipole_save)
                Nph_save_dat = np.zeros_like(Nph_save)
                Nphzp_save_dat = np.zeros_like(Nphzp_save)
                E2_save_dat = np.zeros_like(E2_save)
                E2zp_save_dat = np.zeros_like(E2zp_save)
                E0Etall_save_dat = np.zeros_like(E0Etall_save)
                Q0Qtall_save_dat = np.zeros_like(Q0Qtall_save)
                Penddat = Pend
                Penddatzp = Pendzp
                Qenddat = Qend
                Qenddatzp = Qendzp
                wavefnenddat = wavefnend
            else:
                Penddat = np.hstack([Penddat, Pend])
                Qenddat = np.hstack([Qenddat, Qend])
                wavefnenddat = np.hstack([wavefnenddat, wavefnend])
                Penddatzp = np.hstack([Penddatzp, Pendzp])
                Qenddatzp = np.hstack([Qenddatzp, Qendzp])
            simEdat += simE
            simRhoAdat += simRhoA
            simRhoAdat_adb += simRhoA_adb
            dipole_save_dat += dipole_save
            Nph_save_dat += Nph_save
            Nphzp_save_dat += Nphzp_save
            E2_save_dat += E2_save
            E2zp_save_dat += E2zp_save
            E0Etall_save_dat += E0Etall_save
            Q0Qtall_save_dat += Q0Qtall_save
            r_ind += 1
    if path.exists(calcdir + '/E.csv'):
        simEdat += np.loadtxt(calcdir + '/E.csv', delimiter=',')
    if path.exists(calcdir + '/rho.csv'):
        simRhoAdat += np.loadtxt(calcdir + '/rho.csv', delimiter=',')
    if path.exists(calcdir + '/rho_adb.csv'):
        simRhoAdat_adb += np.loadtxt(calcdir + '/rho_adb.csv', delimiter=',')
    if path.exists(calcdir + '/dipole.npz'):
        dipole_save_dat += np.load(calcdir + '/dipole.npz')['dipole']
    if path.exists(calcdir + '/Nph.npz'):
        Nph_save_dat += np.load(calcdir + '/Nph.npz')['Nph']
    if path.exists(calcdir + '/Nphzp.npz'):
        Nphzp_save_dat += np.load(calcdir + '/Nphzp.npz')['Nphzp']
    if path.exists(calcdir + '/E2.npz'):
        E2_save_dat += np.load(calcdir + '/E2.npz')['E2']
    if path.exists(calcdir + '/E2zp.npz'):
        E2zp_save_dat += np.load(calcdir + '/E2zp.npz')['E2zp']
    if path.exists(calcdir + '/E0Etall.npz'):
        E0Etall_save_dat += np.load(calcdir + '/E0Etall.npz')['E0Etall']
    if path.exists(calcdir + '/Q0Qtall.npz'):
        Q0Qtall_save_dat += np.load(calcdir + '/Q0Qtall.npz')['Q0Qtall']
    if path.exists(calcdir + '/Pend.npz'):
        Penddat = np.hstack([Penddat, np.load(calcdir + '/Pend.npz')['Pend']])
    if path.exists(calcdir + '/Pendzp.npz'):
        Penddat = np.hstack([Penddatzp, np.load(calcdir + '/Pendzp.npz')['Pendzp']])
    if path.exists(calcdir + '/Qend.npz'):
        Qenddat = np.hstack([Qenddat, np.load(calcdir + '/Qend.npz')['Qend']])
    if path.exists(calcdir + '/Qendzp.npz'):
        Qenddatzp = np.hstack([Qenddatzp, np.load(calcdir + '/Qendzp.npz')['Qendzp']])
    if path.exists(calcdir + '/wavefnend.npz'):
        wavefnenddat = np.hstack([wavefnenddat, np.load(calcdir + '/wavefnend.npz')['wavefnend']])
    return simRhoAdat, simRhoAdat_adb, simEdat, dipole_save_dat, Nph_save_dat, Nphzp_save_dat, \
        E2_save_dat, E2zp_save_dat, E0Etall_save_dat, Q0Qtall_save_dat, \
        wavefnenddat, Penddat, Qenddat, Penddatzp, Qenddatzp

def runCalc():
    print('Starting Calculation MF-QED with ' + str(num_atom) + ' atoms')
    print('Using '+str(proc)+' processors with '+str(num_trj)+' trajectories on each processor')
    print('The number of total trajectories is '+str(total_trj)+', which will be executed in '+str(int(total_trj/(proc*num_trj)))+' runs')
    start_time = time.time()
    resRhoA, resRhoA_adb, resE, dipole_save_dat, Nph_save_dat, Nphzp_save_dat, \
        E2_save_dat, E2zp_save_dat, E0Etall_save_dat, Q0Qtall_save_dat,\
            wavefnend, Pend, Qend, Pendzp, Qendzp = parallel_run_ray(total_trj, proc)
    np.savetxt(calcdir + '/E.csv', resE, delimiter=',')
    np.savetxt(calcdir + '/rho.csv', resRhoA, delimiter=',')
    np.savetxt(calcdir + '/rho_adb.csv', resRhoA_adb, delimiter=',')
    np.savez_compressed(calcdir + '/dipole.npz', dipole=dipole_save_dat)
    np.savez_compressed(calcdir + '/Nph.npz', Nph=Nph_save_dat)
    np.savez_compressed(calcdir + '/Nphzp.npz', Nphzp=Nphzp_save_dat)
    np.savez_compressed(calcdir + '/E2.npz', E2=E2_save_dat)
    np.savez_compressed(calcdir + '/E2zp.npz', E2zp=E2zp_save_dat)
    np.savez_compressed(calcdir + '/E0Etall.npz', E0Etall=E0Etall_save_dat)
    np.savez_compressed(calcdir + '/Q0Qtall.npz', Q0Qtall=Q0Qtall_save_dat)
    np.savez_compressed(calcdir + '/Pend.npz', Pend=Pend)
    np.savez_compressed(calcdir + '/Pendzp.npz', Pendzp=Pendzp)
    np.savez_compressed(calcdir + '/Qend.npz', Qend=Qend)
    np.savez_compressed(calcdir + '/Qendzp.npz', Qendzp=Qendzp)
    np.savez_compressed(calcdir + '/wavefnend.npz', wavefnend=wavefnend)
    end_time = time.time()
    print('Finished. Running Wall Time = %10.3f second' % (end_time - start_time))
    return
