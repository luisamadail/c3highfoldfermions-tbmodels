from .auxc3 import *
from .htest import *
from .plotbs import *

import numpy as np
import scipy.linalg as spy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import exp
import sympy as sym
from sympy import *
from IPython.display import display

#########################################

def plot_wave_y_orb(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,nindex,kxval=0):
    #nindex=dimercells*(na+nb-2)//4-1
    
    col=['.b-','.m-','.k-','.r-','.g-','.c-']
    col2=['.b--','.m--','.k--','.r--','.g--','.c--']    
    
    m4=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,kxval)
    m3=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kxval)
    m5=m4+m3
    ener1,wave1=np.linalg.eigh(m5)
    plt.figure(figsize=(3,3))
    plt.plot(ener1,'k.')
    plt.plot(nindex,ener1[nindex],'r.')
    plt.figure()
    for i in range(na+nb-2):
        if i>(na-2):
            plt.plot(np.abs(wave1[i::na+nb-2,nindex]),col2[i-(na-1)],label='A-type ZM')
            print(i)
        else:
            plt.plot(np.abs(wave1[i::na+nb-2,nindex]),col[i],label='B-type ZM')
            

    plt.xlabel("j unit cell index (y-direction)")
    plt.ylabel("$|\psi_j|^2$")
    plt.legend()
    plt.figure()
    return 


# Step 2: Define Fermi-Dirac function
def fermi(E, mu, T): #if temperature dif 0
    return 1 / (np.exp((E - mu) / (k_B * T)) + 1)

def fermi2(E, mu,T): #if temperature equals 0
    return np.heaviside(mu - E, 1.0)  # 1 if E < μ, 0 if E > μ


# Step 3: Solve for chemical potential μ using bisection
def find_mu_bisect(Evals, N_target, T, k_B, tol=1e-6):
    mu_min, mu_max = np.min(Evals), np.max(Evals)
    while mu_max - mu_min > tol:
        mu_mid = 0.5 * (mu_min + mu_max)
        N_occ = np.sum(fermi2(Evals, mu_mid, T))
        if N_occ > N_target:
            mu_max = mu_mid
        else:
            mu_min = mu_mid
    return 0.5 * (mu_min + mu_max)

def plot_charge_yresolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):

    # Lz operator 
    kpoints1=10
    kxvec=np.linspace(-np.pi,np.pi, num=kpoints1)



    T = 0 #0.05 Temperature
    k_B = 1.0

    num_bands = (na+nb-2) * dimercells  # Assuming 4 orbitals per 2-site unit cell

    # Step 1: Collect all eigenvalues

    all_eigvals = []
    a1 = np.array([np.sqrt(3)/2, 1/2])
    a2 = np.array([-np.sqrt(3)/2, 1/2])


    area = np.cross(a1, a2)
    b1 = 2*np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2*np.pi * np.array([-a1[1], a1[0]]) / area

    K = (b1 - b2) / 3

    kxvecK=np.linspace(K[0]*(1-0.3),K[0]*(1),10)
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvecK:
        m31 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, _ = np.linalg.eigh(m30+m31)
        all_eigvals.extend(eigvals)


    all_eigvals = np.array(all_eigvals)
    N_total_states = len(all_eigvals)
    N_target = N_total_states // 2  # Half-filling


    

    # Step 4: Compute chemical potential
    mu = find_mu_bisect(all_eigvals, N_target, T, k_B)
    


    # Step 5: Recalculate Lz using updated μ
    Lz_total = np.zeros(dimercells)
    Lz_per_state = []

    Q_total = np.zeros(dimercells)  # One value per unit cell (4 orbitals per cell -> 2 sites)
    Q_per_state = []
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvec[1:]:
        m3 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, eigvecs = np.linalg.eigh(m30+m3)


        Q_kx_states = np.zeros((len(eigvals), dimercells))

        for n in range(len(eigvals)):
            psi = eigvecs[:, n]
            f_occ = fermi2(eigvals[n], mu, T)

            for j in range(dimercells):
                idx = (na+nb-2) * j
                psi_j = psi[idx:idx+(na+nb-2)]

                charge_density = np.sum(np.abs(psi_j)**2)  # Total probability on site j
                Q_kx_states[n, j] = charge_density
                Q_total[j] += f_occ * charge_density

        Q_per_state.append(Q_kx_states)

    # Normalize total over k-points

    Q_total /= (kpoints1-1)

    print("Total charge/dimer:",sum(Q_total)/dimercells)
    plt.plot(Q_total, 'k.-')
    plt.xlabel("dimer index (j)")
    plt.ylabel(" Q(j)")
    plt.grid(True)
    plt.show()
    return

def compute_Lz_yresolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):

    # Lz operator 
    kpoints1=30
    kxvec=np.linspace(-np.pi,np.pi, num=kpoints1,endpoint=False)


    Lz_op=orb_pert(nb,na,1)
    Lz_opbig=orbperturbation(na,nb,1,dimercells*2)

    T = 0 #0.05 Temperature
    k_B = 1.0

    num_bands = (na+nb-2) * dimercells  # Assuming 4 orbitals per 2-site unit cell

    # Step 1: Collect all eigenvalues
    
    all_eigvals = []
    a1 = np.array([np.sqrt(3)/2, 1/2])
    a2 = np.array([-np.sqrt(3)/2, 1/2])


    area = np.cross(a1, a2)
    b1 = 2*np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2*np.pi * np.array([-a1[1], a1[0]]) / area

    K = (b1 - b2) / 3

    kxvecK=np.linspace(K[0]*(1-0.3),K[0]*(1),10)
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvecK:
        m31 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, _ = np.linalg.eigh(m30+m31)
        all_eigvals.extend(eigvals)


    all_eigvals = np.array(all_eigvals)
    N_total_states = len(all_eigvals)
    N_target = N_total_states // 2  # Half-filling





    # Step 4: Compute chemical potential
    mu = find_mu_bisect(all_eigvals, N_target, T, k_B)
    


    # Step 5: Recalculate Lz using updated μ
    Lz_total = np.zeros(dimercells)
    Lz_per_state = []
    
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvec:
        m3 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, eigvecs = np.linalg.eigh(m30+m3)

        Lz_kx_states = np.zeros((len(eigvals), dimercells))

        for n in range(len(eigvals)):
            psi = eigvecs[:, n]
            f_occ = fermi2(eigvals[n], mu, T)

            for j in range(dimercells):
                idx = (na+nb-2) * j
                psi_j = psi[idx:idx+(na+nb-2)]
                Lz_val = np.real_if_close(np.dot(np.conjugate(psi_j), np.dot(Lz_op, psi_j)), tol=100)
                #Lz_kx_states[n, j] = Lz_val
                Lz_total[j] += f_occ * Lz_val
        #Lz_per_state.append(Lz_kx_states)

    # Normalize total over k-points
    Lz_total /= (kpoints1)
    
    return Lz_total

def plot_Lz_yresolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):

    # Lz operator 
    kpoints1=25
    kxvec=np.linspace(-np.pi,np.pi, num=kpoints1,endpoint=False)


    Lz_op=orb_pert(nb,na,1)
    Lz_opbig=orbperturbation(na,nb,1,dimercells*2)

    T = 0 #0.05 Temperature
    k_B = 1.0

    num_bands = (na+nb-2) * dimercells  # Assuming 4 orbitals per 2-site unit cell

    # Step 1: Collect all eigenvalues

    all_eigvals = []
    a1 = np.array([np.sqrt(3)/2, 1/2])
    a2 = np.array([-np.sqrt(3)/2, 1/2])


    area = np.cross(a1, a2)
    b1 = 2*np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2*np.pi * np.array([-a1[1], a1[0]]) / area

    K = (b1 - b2) / 3

    kxvecK=np.linspace(K[0]*(1-0.3),K[0]*(1),10)
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvecK:
        m31 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, _ = np.linalg.eigh(m30+m31)
        all_eigvals.extend(eigvals)


    all_eigvals = np.array(all_eigvals)
    N_total_states = len(all_eigvals)
    N_target = N_total_states // 2  # Half-filling



    # Step 4: Compute chemical potential
    mu = find_mu_bisect(all_eigvals, N_target, T, k_B)
    


    # Step 5: Recalculate Lz using updated μ
    Lz_total = np.zeros(dimercells)
    Lz_per_state = []
    
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvec:
        m3 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, eigvecs = np.linalg.eigh(m30+m3)

        Lz_kx_states = np.zeros((len(eigvals), dimercells))
        test=[]
        for n in range(len(eigvals)):
            psi = eigvecs[:, n]
            f_occ = fermi2(eigvals[n], mu, T)
            test.append(f_occ)

            for j in range(dimercells):
                idx = (na+nb-2) * j
                psi_j = psi[idx:idx+(na+nb-2)]
                Lz_val = np.real_if_close(np.dot(np.conjugate(psi_j), np.dot(Lz_op, psi_j)), tol=100)
                #Lz_kx_states[n, j] = Lz_val
                Lz_total[j] += f_occ * Lz_val
        #Lz_per_state.append(Lz_kx_states)
    print(sum(test))
    # Normalize total over k-points
    Lz_total /= (kpoints1)
 
    print('Total $L_z$/dimer:',sum(Lz_total)/dimercells)
    plt.plot(Lz_total, 'k.-')
    plt.xlabel("dimer index (j)")
    plt.ylabel("$L_z(j)$")
    plt.grid(True)
    plt.show()
    return

def compute_Lz_band_y_resolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):

    # Lz operator 
    kpoints1=30
    kxvec=np.linspace(-np.pi,np.pi, num=kpoints1)


    Lz_op=orb_pert(nb,na,1)
    Lz_opbig=orbperturbation(na,nb,1,dimercells*2)

    T = 0 #0.05 Temperature
    k_B = 1.0

    num_bands = (na+nb-2) * dimercells  # Assuming 4 orbitals per 2-site unit cell

    # Step 1: Collect all eigenvalues

    all_eigvals = []
    a1 = np.array([np.sqrt(3)/2, 1/2])
    a2 = np.array([-np.sqrt(3)/2, 1/2])


    area = np.cross(a1, a2)
    b1 = 2*np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2*np.pi * np.array([-a1[1], a1[0]]) / area

    K = (b1 - b2) / 3

    kxvecK=np.linspace(K[0]*(1-0.3),K[0]*(1),10)
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvecK:
        m31 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, _ = np.linalg.eigh(m30+m31)
        all_eigvals.extend(eigvals)


    all_eigvals = np.array(all_eigvals)
    N_total_states = len(all_eigvals)
    N_target = N_total_states // 2  # Half-filling





    # Step 4: Compute chemical potential
    mu = find_mu_bisect(all_eigvals, N_target, T, k_B)
    


    # Step 5: Recalculate Lz using updated μ
    Lz_total = np.zeros(dimercells)
    Lz_per_state = []
    
    m30=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    for kx in kxvec[1:]:
        m3 = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx)
        eigvals, eigvecs = np.linalg.eigh(m30+m3)

        Lz_kx_states = np.zeros((len(eigvals), dimercells))

        for n in range(len(eigvals)):
            psi = eigvecs[:, n]
            f_occ = fermi2(eigvals[n], mu, T)

            for j in range(dimercells):
                idx = (na+nb-2) * j
                psi_j = psi[idx:idx+(na+nb-2)]
                Lz_val = np.real_if_close(np.dot(np.conjugate(psi_j), np.dot(Lz_op, psi_j)), tol=100)
                Lz_kx_states[n, j] = Lz_val
                #Lz_total[j] += f_occ * Lz_val

        Lz_per_state.append(Lz_kx_states)

    return Lz_per_state

def plot_Lz_band_k_yresolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    
    Lz_per_state=compute_Lz_band_y_resolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield)
    nmono=dimercells*2
    y_vals = np.arange(int(nmono/2))

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    nindex=[nmono-1,nmono]
    ener1,wave1=np.linalg.eigh(h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,np.pi)+h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,np.pi))
    ax[0].plot(ener1,'k.')
    ax[0].plot(nindex,ener1[nindex],'r.')
    ax[0].set_title("energies at $k=\pi$")
    ax[0].set_ylabel("E/t")
    ax[0].set_xlabel("index")
    ener1,wave1=np.linalg.eigh(h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,2*np.pi/3)+h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,2*np.pi/3))
    ax[1].plot(ener1,'k.')
    ax[1].plot(nindex,ener1[nindex],'r.')
    ax[1].set_title("energies at $k=2\pi/3$")
    ax[1].set_xlabel("index")
    ener1,wave1=np.linalg.eigh(h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)+h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,0))
    ax[2].plot(ener1,'k.')
    ax[2].plot(nindex,ener1[nindex],'r.')
    ax[2].set_title("energies at $k=0$")
    ax[2].set_xlabel("index")
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    col2=['b','m','k','r','g','c']
   
    # Plot a few eigenstates (kx = 0)
    for i, band_idx in enumerate([nmono - 1, nmono]):
        Lz_example1 = Lz_per_state[0][band_idx]
        ax[0,0].plot(y_vals, Lz_example1,col2[i], label=f"State {band_idx}")
        Lz_example2 = Lz_per_state[len(Lz_per_state)//6][band_idx]
        ax[0,1].plot(y_vals, Lz_example2,col2[i], label=f"State {band_idx}")
        Lz_example3 = Lz_per_state[len(Lz_per_state)//2-1][band_idx]
        ax[1,0].plot(y_vals, Lz_example3,col2[i], label=f"State {band_idx}")

    ax[0,0].set_title(r"$\langle L_z \rangle_j$ per eigenstate at $k_x=\pi$")
    ax[0,0].set_xlabel("Site index $j$ (y-direction)")
    ax[0,0].set_ylabel(r"$\langle L_z \rangle$")
    ax[0,0].legend()
    ax[0,0].grid(True)

    ax[0,1].set_title(r"$\langle L_z \rangle_j$ per eigenstate at fixed $k_x=2\pi/3$")
    ax[0,1].set_xlabel("Site index $j$ (y-direction)")
    ax[0,1].set_ylabel(r"$\langle L_z \rangle$")
    ax[0,1].legend()
    ax[0,1].grid(True)

    ax[1,0].set_title(r"$\langle L_z \rangle_j$ per eigenstate at fixed $k_x=0$")
    ax[1,0].set_xlabel("Site index $j$ (y-direction)")
    ax[1,0].set_ylabel(r"$\langle L_z \rangle$")
    ax[1,0].legend()
    ax[1,0].grid(True)

    Lz_total=compute_Lz_yresolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield)
    # Total Lz
    ax[1,1].plot(y_vals, Lz_total, color='black')
    ax[1,1].set_title(r"Total $\langle L_z \rangle_j$ (Fermi-Dirac occupied)")
    ax[1,1].set_xlabel("Site index $j$ (y-direction)")
    ax[1,1].set_ylabel(r"$\langle L_z \rangle$")
    ax[1,1].grid(True)

    plt.tight_layout()
    plt.show()
    return

def compute_Lz_Efield(na,nb,a,t1,t3,hamiltoniantype,dimercells,mataux,orbpert_strength,sublpert_strength,Bfield,E):
    
    Efield=E
    
    T = 0 #0.05 Temperature
    k_B = 1.0
    kpoints1=20
    kxvec=np.linspace(-np.pi,np.pi, num=kpoints1)
    

    # Step 2: Define Fermi-Dirac function


    
    nmono=dimercells*2
    Lz_first = 0.0
    Lz_last = 0.0
    Lz_mid = 0.0
    j1 = 0
    j2 = int(nmono/2)-1
    j3 = int(nmono/2/2)
    idxfirst = (na+nb-2) * j1
    idxlast = (na+nb-2) * j2
    idxmid = (na+nb-2) * j3

    m30=Hmultiorbital_ZZribbon_red(0,na,nb,mataux,2*dimercells,0,t1,t3,Bfield,Efield,orbpert_strength,sublpert_strength)
    m30+=Electricf_T(na,nb,a,2*dimercells,Efield)
    
    for kx in kxvec:
        m3=Hmultiorbital_ZZribbon_red(1,na,nb,mataux,2*dimercells,kx,t1,t3,Bfield,Efield,orbpert_strength,sublpert_strength)
        eigvals, eigvecs = np.linalg.eigh(m30+m3)

        Lz_op = orb_pert(nb,na,1)

        for n in range(len(eigvals)//2):
            psi = eigvecs[:, n]

            psi_first = psi[idxfirst:idxfirst+(na+nb-2)]
            psi_last = psi[idxlast:idxlast+(na+nb-2)]
            psi_mid = psi[idxmid:idxmid+(na+nb-2)]

            Lz_valfirst = np.real_if_close(np.dot(np.conjugate(psi_first), np.dot(Lz_op, psi_first)), tol=100)
            Lz_vallast = np.real_if_close(np.dot(np.conjugate(psi_last), np.dot(Lz_op, psi_last)), tol=100)
            Lz_valmid = np.real_if_close(np.dot(np.conjugate(psi_mid), np.dot(Lz_op, psi_mid)), tol=100)

            Lz_first += Lz_valfirst
            Lz_last += Lz_vallast
            Lz_mid += Lz_valmid

    return Lz_first / kpoints1, Lz_last / kpoints1, Lz_mid / kpoints1


def plot_LzvsEfield(na,nb,a,t1,t3,hamiltoniantype,dimercells,mataux,orbpert_strength,sublpert_strength,Bfield,Efield):
    
    from matplotlib.ticker import MaxNLocator
    from joblib import Parallel, delayed


    Efvec=np.linspace(-0.0000001,0.0000001,51)
    results1 = Parallel(n_jobs=-1)(delayed(compute_Lz_Efield)(na,nb,a,t1,t3,hamiltoniantype,dimercells,mataux,orbpert_strength,sublpert_strength,Bfield,E) for E in Efvec)
    Lz_totalfirst1, Lz_totallast1, Lz_totalmid1 = zip(*results1)



    plt.figure(figsize=(7, 9))

    # Plot curves
    plt.plot(Efvec, Lz_totalfirst1, 'r.-', label='j - bottom edge ($\phi=0$)', markersize=5)
    plt.plot(Efvec, Lz_totallast1, 'b.-', label='j - top edge ($\phi=0$)', markersize=5)
    plt.plot(Efvec, Lz_totalmid1, 'k.-', label='j - middle ($\phi=0$)', markersize=5)

    # Axes labels and title
    plt.xlabel("$E_{field}$", fontsize=15)
    plt.ylabel(r"$\langle L_z \rangle$", fontsize=15)
    plt.title(r"Total $\langle L_z \rangle_j$ vs $E_{field}$", fontsize=15)

    # Reduce x-axis tick density
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  # adjust nbins as needed

    # Formatting

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return
