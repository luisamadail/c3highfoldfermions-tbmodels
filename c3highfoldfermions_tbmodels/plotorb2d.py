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


def get_reciprocal_lattice(a1, a2):
    area = np.cross(a1, a2)
    b1 = 2 * np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2 * np.pi * np.array([-a1[1], a1[0]]) / area
    return b1, b2


def dH_dk(kx, ky,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,delta=1e-5):
    dHdkx = (h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx + delta, ky) - h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx - delta, ky)) / (2 * delta)
    dHdky = (h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, ky + delta) - h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, ky - delta)) / (2 * delta)
    return dHdkx, dHdky


def dH_dk2(kx, ky,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,delta=1e-5):
    
    
    dHdkx = (HTB_2D(na,nb,kx+delta,ky,orbpert_strength,sublpert_strength) - HTB_2D(na,nb,kx-delta,ky,orbpert_strength,sublpert_strength)) / (2 * delta)
    dHdky = (HTB_2D(na,nb,kx,ky+delta,orbpert_strength,sublpert_strength) - HTB_2D(na,nb,kx,ky-delta,orbpert_strength,sublpert_strength)) / (2 * delta)
    return dHdkx, dHdky


# Berry curvature
def berry_total(kx, ky,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield):
    num_occupied=(na + nb - 2) // 2
    H = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, ky)
    dHdkx, dHdky = dH_dk(kx, ky,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)
    evals, evecs = np.linalg.eig(H)
    idx = np.argsort(evals.real)
    evals = evals[idx].real
    evecs = evecs[:, idx]
    N = len(evals)
    omega_total = 0.0
    for n in range(num_occupied):
        for m in range(N):
            if m == n:
                continue
            En, Em = evals[n], evals[m]
            ket_n = evecs[:, n]
            ket_m = evecs[:, m]
            v1 = np.vdot(ket_n, dHdkx @ ket_m)
            v2 = np.vdot(ket_m, dHdky @ ket_n)
            omega_total += -2 * np.imag(v1 * v2) / (En - Em)**2
    return omega_total

def oam(kx,ky):
    ham=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx,ky)
    #k = np.array([kx,ky])
    ener,wave=np.linalg.eigh(ham) #wave[site,k]
    L=orb_pert(na,nb,1)
    Lexp=np.zeros(na+nb-2)
    for i in range(na+nb-2):
        Lexp[i]=np.real_if_close(np.dot(np.conjugate(wave[:,i]),np.dot(L,wave[:,i])),tol=100)
    return Lexp,ener


def fermi2(E, mu,T): #if temperature equals 0
    return np.heaviside(mu - E, 1.0)  # 1 if E < μ, 0 if E > μ

def oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,kx,ky,return_energies=False):
    ham=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx,ky)
    ener,wave=np.linalg.eigh(ham)
    Lz = orb_pert(na,nb,1)
    Lexp=np.zeros(len(ener))
    for i in range(len(ener)):
        Lexp[i] = np.real(np.dot(np.conj(wave[:, i]), np.dot(Lz, wave[:, i])))
    if return_energies:
        return Lexp, ener
    return Lexp

def plot_OAM_kresolved_all(na,nb,t1,t3,a,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    kxvec=np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),401,endpoint=False)
    plt.figure(figsize=(6,4))
    col=['.-b','.-m','.-k','.-r','.-g','.-c']
    col2=['b','m','k','r','g','c']
    col3=['b--','m--','k--','r--','g--','c--']
    ekzz=np.zeros((len(kxvec),na+nb-2))
    for i in range(na+nb-2):
        for j in range(len(kxvec)):

            ekzz[j,:]=np.real_if_close(oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,kxvec[j],0,return_energies=True)[1], tol=100)

        plt.plot(kxvec,ekzz[:,i],col2[i],linewidth=1.4)
    plt.grid(True)
    plt.xlabel('$\phi_2=-\phi_1$')
    plt.ylabel('$E/t$')

    plt.figure(figsize=(6,4))

    oam_k=np.zeros((na+nb-2,len(kxvec)))

    for i in range(na-1):
        for j in range(len(kxvec)):

            oam_k[i,j]=oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,kxvec[j],0, return_energies=True)[0][i]
        plt.plot(kxvec,(oam_k[i,:]),col2[i])
    plt.grid(True)
    plt.xlabel('$\phi_1$')
    plt.ylabel('$L(\phi_1)$')
        #plt.xlim(-0.1,0.1)

    plt.figure(figsize=(6,4))
    for i in range(nb-1):
        for j in range(len(kxvec)):



            oam_k[na+i-1,j]=oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,kxvec[j],0, return_energies=True)[0][na+i-1]
        plt.plot(kxvec,(oam_k[na+i-1,:]),col2[na+i-1])
    plt.grid(True)
    plt.xlabel('$\phi_1$')
    plt.ylabel('$L(\phi_1)$')
        #plt.xlim(-0.1,0.1)


        #plt.xlim(-0.1,0.1)

    L_total_k = np.zeros_like(kxvec)

    plt.figure(figsize=(6,4))

    for j, kx in enumerate(kxvec):
        Lexp, energies = oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,kx, 0, return_energies=True)
        occs = fermi2(energies, mu=0.0, T=300)
        L_total_k[j] = np.sum(occs * Lexp)

    plt.plot(kxvec,L_total_k,'k')
    for i in range(na-1):
        plt.plot(kxvec,(oam_k[i,:]),col3[i],linewidth=0.7)
    plt.grid(True)
    plt.xlabel('$\phi_1$')
    plt.ylabel('$L_{total}(\phi_1)$')
    return

def plot_OAM_kresolved_total(na,nb,t1,t3,a,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    
    from IPython.display import display, Math

    kxvec=np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),401,endpoint=False)
    ekzz = np.zeros((len(kxvec), na + nb - 2))
    L_total_k = np.zeros_like(kxvec)
    display(Math("L_z ="))
    print(orb_pert(na,nb,1))    
    print('orbital pert. :') 
    print(orb_pert(na,nb,orbpert_strength))
    print('sublattice pert. : ')
    print(sub_pert(na,nb,sublpert_strength))
    # Compute energy bands and orbital moment
    for j, kx in enumerate(kxvec):
        Lexp, energies = oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,kx, 0, return_energies=True)
        occs = fermi2(energies, mu=0.0, T=300)
        L_total_k[j] = np.sum(occs * Lexp)
        ekzz[j, :] = np.real_if_close(energies[:na + nb - 2], tol=100)

    # Start combined plot
    fig, ax1 = plt.subplots(figsize=(6,4))

    # Left y-axis: energy bands
    colors = ['k', 'b', 'r', 'g']
    for i in range(na + nb - 2):
        ax1.plot(kxvec, ekzz[:, i], 'k--', linewidth=1.4)
    ax1.set_xlabel(r'$\phi_2 = -\phi_1$', fontsize=12)
    ax1.set_ylabel(r'$E/t$', fontsize=12, color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    
    a1 = np.array([np.sqrt(3)/2, 1/2])*(na+nb+1)*a
    a2 = np.array([-np.sqrt(3)/2, 1/2])*(na+nb+1)*a

    area = np.cross(a1, a2)
    b1 = 2*np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2*np.pi * np.array([-a1[1], a1[0]]) / area

    K = (b1 - b2) / 3
    Kp = (b2 - b1) / 3



    ticks = [Kp[0], 0, K[0]]
    labels = [r"$K'$", r"$\Gamma$", r"$K$"]

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)



    # Right y-axis: orbital moment
    ax2 = ax1.twinx()
    ax2.plot(kxvec, L_total_k, 'r', linewidth=1.5, label=r'$L_{\mathrm{total}}$')
    ax2.set_ylabel(r'$L_{\mathrm{total}}(\phi_1)$', fontsize=12, color='k')

    ax2.tick_params(axis='y', labelcolor='k')

    plt.title('Energy bands and total OAM', fontsize=14)
    plt.show()
    return

def print_OAM_total_BZ(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,npoints=300):
    phi1 = np.linspace(-np.pi/((na+nb+1)*a), np.pi/((na+nb+1)*a), npoints, endpoint=False)  # adjust 100 for resolution
    phi2 = np.linspace(-np.pi/((na+nb+1)*a), np.pi/((na+nb+1)*a), npoints, endpoint=False)
    PHI1, PHI2 = np.meshgrid(phi1, phi2)

    L_total_sum = 0.0
    num_kpoints = PHI1.size  # 100*100 = 10000

    for i in range(len(phi1)):
        for j in range(len(phi2)):
            Lexp, energies = oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,PHI1[j,i], PHI2[j,i], return_energies=True)
            occs = fermi2(energies, mu=0.0, T=300)  # or use fermi_dirac for finite T
            L_total_sum += np.sum(occs * Lexp)

    L_total = L_total_sum / (num_kpoints)
    print("Total OAM over BZ:", L_total)
    return

def plot_BerryC_kresolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    
    Nk = 101
    kx_vals = np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),Nk,endpoint=False)
    Omega_total = np.zeros(Nk)
    num_occupied=(na + nb - 2) // 2
    dk = kx_vals[1] - kx_vals[0]
    
    for i, kx in enumerate(kx_vals):
        H = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, 0)
        dHdkx, dHdky = dH_dk(kx, 0,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)
        evals, evecs = np.linalg.eigh(H)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        omega_sum = 0.0
        for band_index in range(num_occupied):
            omega_n = 0.0
            for m in range(len(evals)):
                if m == band_index:
                    continue
                En, Em = evals[band_index], evals[m]
                ket_n = evecs[:, band_index]
                ket_m = evecs[:, m]
                v1 = np.vdot(ket_n, dHdkx @ ket_m)
                v2 = np.vdot(ket_m, dHdky @ ket_n)
                omega_n += -2 * np.imag(v1 * v2) / (En - Em)**2
            omega_sum += omega_n

        Omega_total[i] = omega_sum / (2 * np.pi)

    
    partial_chern_total = np.trapz(Omega_total, kx_vals)
    
    plt.plot(kx_vals, Omega_total,'k', label='Sum over occupied bands')
    plt.title(r'Total Berry curvature along $ky=0$ for occupied bands')
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$\sum_{n \in occ} \Omega_n(k_x, 0)$')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print(f'Partial Chern number slice at k_y=0 (all occupied bands): {partial_chern_total:.4f}')

    return

def print_ChernN_band(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):

    Nk=101
    kx_vals = np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),Nk,endpoint=False)
    ky_vals = np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),Nk,endpoint=False)
    dk = (kx_vals[1] - kx_vals[0]) * (ky_vals[1] - ky_vals[0])

    num_occupied=(na + nb - 2) // 2


    chern_per_band = np.zeros(4)  # For 4 bands

    for band_index in range(na+nb-2):
        Omega_band = np.zeros((Nk, Nk))  # shape (201, 201)
        for i, kx in enumerate(kx_vals):
            for j, ky in enumerate(ky_vals):
                H = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, 0)
                dHdkx, dHdky = dH_dk(kx, 0,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)
                evals, evecs = np.linalg.eigh(H)
                idx = np.argsort(evals)
                evals = evals[idx]
                evecs = evecs[:, idx]

                omega_n = 0.0
                for m in range(na+nb-2):
                    if m == band_index:
                        continue
                    En, Em = evals[band_index], evals[m]
                    ket_n = evecs[:, band_index]
                    ket_m = evecs[:, m]
                    v1 = np.vdot(ket_n, dHdkx @ ket_m)
                    v2 = np.vdot(ket_m, dHdky @ ket_n)
                    omega_n += -2 * np.imag(v1 * v2) / (En - Em)**2

                # Safety check for indices (should never trigger)
                if i >= Nk or j >= Nk:
                    print(f"Index out of bounds: i={i}, j={j}, Nk={Nk}")
                Omega_band[i, j] = omega_n / (2 * np.pi) 

        chern_per_band[band_index] = np.sum(Omega_band) * dk 

    print("Chern numbers per band:", chern_per_band)
    return



def plot_BerryC_BZ(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    
    import matplotlib.colors as mcolors
    
    # --- Brillouin zone grid ---
    a0 = 2.42  # lattice constant in Angstrom
    R = na + nb + 1
    #a2_real = R * a0 * np.array([np.sqrt(3)/2,0.5])
    a1_real = R * a0 * np.array([np.sqrt(3)/2,0.5])
    a2_real = R * a0 * np.array([-np.sqrt(3)/2,0.5])

    b1, b2 = get_reciprocal_lattice(a1_real, a2_real)
    
    Nk = 101
    i_vals = np.linspace(-0.5, 0.5, Nk,endpoint=False)
    j_vals = np.linspace(-0.5, 0.5, Nk,endpoint=False)
    I, J = np.meshgrid(i_vals, j_vals, indexing='ij')

    kx_vals = I * b1[0] + J * b2[0]
    ky_vals = I * b1[1] + J * b2[1]
    Omega_grid = np.zeros((Nk, Nk))

    # --- Compute Berry curvature ---
    print("Computing Berry curvature grid...")
    for i in range(Nk):
        for j in range(Nk):
            Omega_grid[i, j] = berry_total(kx_vals[i, j]/np.sqrt(3)*2, ky_vals[i, j]*2,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)
            #Omega_grid[i, j] = berry_total(kx_vals[i, j], ky_vals[i, j],na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)

    # --- Plotting ---

    # Define custom red-to-white colormap
    seismic = plt.get_cmap('seismic')
    red_half = mcolors.LinearSegmentedColormap.from_list(
        'red_half', seismic(np.linspace(0.5, 1.0, 256))
    )

    if sublpert_strength!=0:
        colmap=seismic
    if orbpert_strength!=0:
        colmap=red_half

    plt.figure(figsize=(6, 5))
    #plt.pcolormesh(kx_vals/np.sqrt(3)*2, ky_vals*2, Omega_grid, shading='auto', cmap=colmap)
    plt.pcolormesh(kx_vals, ky_vals, Omega_grid, shading='auto', cmap=colmap)
    plt.colorbar(label=r'$\Omega_{\mathrm{tot}}(\mathbf{k})$')
    plt.xlabel(r'$\phi_1$')
    plt.ylabel(r'$\phi_2$')
    plt.title('Total Berry curvature of occupied bands')

    # Plot hexagonal BZ boundary
    # Proper BZ hexagon vertices based on reciprocal lattice
    bz_vertices = []
    angles = np.linspace(0, 2*np.pi, 7) # 6 points, omit duplicate
    radius = 1 / 3 * np.linalg.norm(b1 + b2)

    # Get unit vectors in 6 directions and scale by proper radius
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        vertex = radius * direction
        bz_vertices.append(vertex)
    bz_vertices = np.array(bz_vertices)

    # Plot BZ boundary
    plt.plot(bz_vertices[:, 0]/2, bz_vertices[:, 1]*np.sqrt(3)/2, 'k-', lw=1)
    #plt.plot(bz_vertices[:, 0], bz_vertices[:, 1], 'k-', lw=1)


    #plt.plot(bz_vertices[:, 0]*2, bz_vertices[:, 1]/np.sqrt(3)*2, 'k-', lw=1)

    plt.tight_layout()
    plt.show()
    return

def BerryC_kpoint(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx,ky):
    H = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, ky)
    dHdkx, dHdky = dH_dk(kx, ky,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)
    evals, evecs = np.linalg.eig(H)
    idx = np.argsort(evals.real)
    evals = evals[idx].real
    evecs = evecs[:, idx]
    N = len(evals)
    omega = np.zeros(N)

    for n in range(N):
        for m in range(N):
            if m == n:
                continue
            En, Em = evals[n], evals[m]
            ket_n = evecs[:, n]
            ket_m = evecs[:, m]
            v1 = np.vdot(ket_n, dHdkx @ ket_m)
            v2 = np.vdot(ket_m, dHdky @ ket_n)
            omega[n] += -2 * np.imag(v1 * v2) / (En - Em)**2
    return omega

def plot_BerryCband_kresolved(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    col2=['b','m','k','r','g','c']
    Nk = 101
    kx_list = np.linspace(-np.pi/((na+nb+1)*a), np.pi/((na+nb+1)*a), Nk,endpoint=False)
    ky = 0.0
    berry_all_bands = np.zeros((Nk, na+nb-2))

    for i, kx in enumerate(kx_list):
        omega_k = BerryC_kpoint(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx, ky)
        berry_all_bands[i, :] = omega_k 

    # Plot
    plt.figure(figsize=(7, 5))
    for band in range(na+nb-2):
        plt.plot(kx_list, berry_all_bands[:, band],col2[band], label=f'Band {band}')
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'Berry curvature $\Omega_n(k_x, k_y=0)$')
    plt.title('Berry curvature along $k_y=0$')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return


def compute_mz_antisym(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx, ky, delta_k, band_index):
    """
    Compute orbital magnetic moment m_n^z at (kx, ky) using antisymmetric semiclassical formula.

    Parameters:
    - kx, ky: float, momentum components (in reciprocal space units)
    - delta_k: float, finite difference step
    - H_func: function H(kx, ky) returning Bloch Hamiltonian (Hermitian ndarray)
    - band_index: int, band index `n`

    Returns:
    - m_z: float, orbital magnetic moment in units of A·m^2
    """
    hbar = 1#1.055e-34  # J·s
    e = 1#1.602e-19     # C
    a = 2.42      # graphene lattice constant (meters)
    
    def get_H_eigensystem(kx, ky):
        H=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, ky)
        evals, evecs = np.linalg.eigh(H)
        return H, evals, evecs

    # Get central Hamiltonian and eigenvectors
    
    H0, evals0, evecs0 = get_H_eigensystem(kx, ky)
    eps_n = evals0[band_index]
    u_n = evecs0[:, band_index]

    # Compute finite difference derivatives of |u_n>
    _, _, u_x_plus = get_H_eigensystem(kx + delta_k, ky)
    _, _, u_x_minus = get_H_eigensystem(kx - delta_k, ky)
    _, _, u_y_plus = get_H_eigensystem(kx, ky + delta_k)
    _, _, u_y_minus = get_H_eigensystem(kx, ky - delta_k)

    # Use phase alignment to minimize numerical noise (overlap gauge)
    def phase_align(u_ref, u_target):
        phase = np.vdot(u_ref, u_target)
        return u_target * np.exp(-1j * np.angle(phase))

    u_xp_n = phase_align(u_n, u_x_plus[:, band_index])
    u_xm_n = phase_align(u_n, u_x_minus[:, band_index])
    u_yp_n = phase_align(u_n, u_y_plus[:, band_index])
    u_ym_n = phase_align(u_n, u_y_minus[:, band_index])

    du_dx = (u_xp_n - u_xm_n) / (2 * delta_k)
    du_dy = (u_yp_n - u_ym_n) / (2 * delta_k)

    # Define the (H - ε_n) operator
    H_shifted = H0 - eps_n * np.eye(H0.shape[0])

    # Compute the matrix elements
    term1 = np.vdot(du_dx, H_shifted @ du_dy)
    term2 = np.vdot(du_dy, H_shifted @ du_dx)

    # Antisymmetric combination
    mz = (e / (2 * hbar)) * (term1 - term2)

    # Normalize to units of e a^2 / hbar
    #mu_B = e * hbar / (2 * 9.10938356e-31)  # e·ħ/2m_e
    scale =  a**2  
    mz /= scale

    return np.imag(mz)  # in A·m^2


def orbital_momentgxiao(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx, ky, delta=1e-5):
    hbar = 1#1.055e-34  # J·s
    e = -1#1.602e-19     # C
    a=2.42
    # Derivatives wrt φ1, φ2 (finite differences)
    H = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield, 1, kx, ky)
    dH_dkx, dH_dky = dH_dk(kx, ky,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)

    # Diagonalize
    energies, eigvecs = np.linalg.eigh(H)
    m_orb = np.zeros(len(energies), dtype='complex_')

    # Xiao formula
    for n in range(len(energies)):
        for m in range(len(energies)):
            if m != n:
                num = (
                    np.vdot(eigvecs[:, n], dH_dkx @ eigvecs[:, m])
                    * np.vdot(eigvecs[:, m], dH_dky @ eigvecs[:, n])
                    - np.vdot(eigvecs[:, n], dH_dky @ eigvecs[:, m])
                    * np.vdot(eigvecs[:, m], dH_dkx @ eigvecs[:, n])
                )
                m_orb[n] += np.imag(num) / (energies[n] - energies[m])
    prefactor = 1/2 *(e/hbar) # (e = ℏ = 1 units)
    scale =  a**2  
    return prefactor * m_orb/scale, energies

def orbital_momentgxiao2(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx, ky,delta=1e-5):
    
    hbar = 1#1.055e-34  # J·s
    e = -1#1.602e-19     # C
    a=2.42
    # Derivatives wrt φ1, φ2 (finite differences)
    
    H = HTB_2D(na,nb,kx,ky,orbpert_strength,sublpert_strength)
    dH_dkx, dH_dky = dH_dk2(kx, ky,na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield)

    # Diagonalize
    energies, eigvecs = np.linalg.eigh(H)
    m_orb = np.zeros(len(energies), dtype='complex_')

    # Xiao formula
    for n in range(len(energies)):
        for m in range(len(energies)):
            if m != n:
                num = (
                    np.vdot(eigvecs[:, n], dH_dkx @ eigvecs[:, m])
                    * np.vdot(eigvecs[:, m], dH_dky @ eigvecs[:, n])
                    - np.vdot(eigvecs[:, n], dH_dky @ eigvecs[:, m])
                    * np.vdot(eigvecs[:, m], dH_dkx @ eigvecs[:, n])
                )
                m_orb[n] += np.imag(num) / (energies[n] - energies[m])
    prefactor = 1/2 *(e/hbar) # (e = ℏ = 1 units)
    scale =  a**2  
    return prefactor * m_orb/scale, energies



def plot_OMM_kresolved_all(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    
    kxvec=np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),201, endpoint=False)
    num_bands = (na+nb-2)
    m_orb_bands = np.zeros((num_bands, len(kxvec)))
    col2=['b','m','k','r','g','c']

    plt.figure(figsize=(6,4))
    
    col3=['b--','m--','k--','r--','g--','c--']
    ekzz=np.zeros((len(kxvec),na+nb-2))
    for i in range(na+nb-2):
        for j in range(len(kxvec)):

            ekzz[j,:]=np.real_if_close(oam2(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,kxvec[j],0,return_energies=True)[1], tol=100)

        plt.plot(kxvec,ekzz[:,i],col2[i],linewidth=1.4)
    plt.grid(True)
    plt.xlabel('$\phi_2=-\phi_1$')
    plt.ylabel('$E/t$')
    
    for j in range(len(kxvec)): 
        for band in range(num_bands):
            m_orb_bands[band, j] = np.real_if_close(compute_mz_antisym(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kxvec[j],0, 0.0001, band),tol=100000000)

    plt.figure()
    for band in range(num_bands):
        plt.plot(kxvec, m_orb_bands[band,:], color=col2[band], label=f'Band {band+1}')
    plt.xlabel('$\phi_1$')
    plt.ylabel(r'$m^z_n(k)\, /\, (e a^2/\hbar)$')
    plt.title('Orbital Magnetic Moment per Band along $\phi_1=-\phi_2$')
    plt.legend()
    #plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    for band in range(na-1):
        plt.plot(kxvec, m_orb_bands[band,:], col3[band], label=f'Band {band+1}',linewidth=0.7)
    
    omm_total = np.zeros_like(kxvec,dtype = 'complex_')
    for j, kx in enumerate(kxvec):
        m_bands, energies = orbital_momentgxiao(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx, 0)
        occs = fermi2(energies, mu=0.0, T=300)
        omm_total[j] = np.sum(occs * m_bands)

    plt.plot(kxvec, np.real_if_close(omm_total,tol=100000000), 'k')
    plt.xlabel('$\phi_1(k)$')
    plt.ylabel('Total $m_{n,k}$ (nat. units)')
    plt.grid(True)
    return


def print_orbital_magnetization(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    Nphi=201
    kx_vals = np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),Nphi, endpoint=False)
    dkx = kx_vals[1] - kx_vals[0]
    
    M = 0.0
    for kx in kx_vals:
        for ky in kx_vals:
            m_bands, energies = orbital_momentgxiao(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx, ky)
            occs = fermi2(energies, mu=0, T=0)
            M += np.sum(occs * np.real_if_close(m_bands,tol=1000000))

    # Brillouin zone area is (2π)^2
    M *= (dkx**2)
    print("Total Magnetization over BZ (natural units):", M)
    
    mu_B = 5.788e-5  # Bohr magneton in eV/T
    mu_B_SI = 9.274e-24  # A·m²

    print("Total Magnetization over BZ (eV/T):", mu_B* M)
    print("Total Magnetization over BZ (A·m²):", mu_B_SI*M)

    return 

def compute_orbital_magnetization(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    Nphi=51
    kx_vals = np.linspace(-np.pi/((na+nb+1)*a),np.pi/((na+nb+1)*a),Nphi, endpoint=False)
    dkx = kx_vals[1] - kx_vals[0]
    
    M = 0.0
    for kx in kx_vals:
        for ky in kx_vals:
            m_bands, energies = orbital_momentgxiao2(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kx, ky)
            occs = fermi2(energies, mu=0, T=0)
            M += np.sum(occs * np.real_if_close(m_bands,tol=1000000))

    # Brillouin zone area is (2π)^2
    M *= (dkx**2)
    
    mu_B = 5.788e-5  # Bohr magneton in eV/T
    mu_B_SI = 9.274e-24  # A·m²

    return M

# Remove degeneracies 

def plot_Mag_vs_orbpert(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):

    a1 = np.array([np.sqrt(3)/2, 1/2])*(na+nb+1)*a
    a2 = np.array([-np.sqrt(3)/2, 1/2])*(na+nb+1)*a

    area = np.cross(a1, a2)
    b1 = 2*np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2*np.pi * np.array([-a1[1], a1[0]]) / area

    K = (b1 - b2) / 3
    Kp = (b2 - b1) / 3

    nbet=41
    betvec0=np.linspace(-0.1,0.1,nbet)
    # Remove degeneracies
    #remove_vals = [-0.054,-0.001, 0.001, 0.054]
    # Use np.isin with ~ to filter them out (with isclose for safety)
    #mask = np.ones_like(betvec0, dtype=bool)
    #for val in remove_vals:
    #    mask &= ~np.isclose(betvec0, val,atol=1e-03)
    #betvec = betvec0[mask]

    col=['.-b','.-m','.-k','.-r','.-g','.-c']
    betvec = betvec0
    magbet=[]
    num_bands=(na+nb-2)

    def energiesbet(kx,ky,bet):
        hambet=HTB_2D(na,nb,kx,ky,bet,sublpert_strength)
        enerbet,wave=np.linalg.eigh(hambet) #wave[site,k]
        return enerbet

    enerbetvec=np.zeros((num_bands,len(betvec)))
    enerbetvec2=np.zeros((num_bands,len(betvec)))
    
    for i in range(len(betvec)):
        magbet.append(compute_orbital_magnetization(na,nb,a,t1,t3,hamiltoniantype,dimercells,betvec[i],sublpert_strength,Bfield,Efield))
        for j in range(num_bands):
            enerbetvec[j,i]=energiesbet(0,0,betvec[i])[j]
            enerbetvec2[j,i]=energiesbet(K[0],K[1],betvec[i])[j]

    for j in range(num_bands):
        plt.plot(betvec,enerbetvec[j,:],col[j])
    plt.ylabel('$E(\u03B1)$')
    plt.xlabel('$\u03B1$')
    plt.title('energy bands at $\phi_1=-\phi_2=0$')
    plt.figure()
    for j in range(num_bands):
        plt.plot(betvec,enerbetvec2[j,:],col[j])
    plt.ylabel('$E(\u03B1)$')
    plt.xlabel('$\u03B1$')
    plt.title('energy bands at $\phi_1=-\phi_2=2\pi/3$')
    plt.figure()
    plt.plot(betvec,magbet,'.-k')
    plt.xlabel('$\u03B1$')
    plt.ylabel('$M(\u03B1)$')
    plt.title('Magnetization dependency on $V_{orb}$') 
    return

def plot_Mag_vs_orbpert_allin1(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):

    a1 = np.array([np.sqrt(3)/2, 1/2])*(na+nb+1)*a
    a2 = np.array([-np.sqrt(3)/2, 1/2])*(na+nb+1)*a

    area = np.cross(a1, a2)
    b1 = 2*np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2*np.pi * np.array([-a1[1], a1[0]]) / area

    K = (b1 - b2) / 3
    Kp = (b2 - b1) / 3

    nbet=40
    betvec0=np.linspace(-0.1,0.1,nbet)
    # Remove degeneracies
    #remove_vals = [-0.054,-0.001, 0.001, 0.054]
    # Use np.isin with ~ to filter them out (with isclose for safety)
    #mask = np.ones_like(betvec0, dtype=bool)
    #for val in remove_vals:
    #    mask &= ~np.isclose(betvec0, val,atol=1e-03)
    #betvec = betvec0[mask]

    col=['.-b','.-m','.-k','.-r','.-g','.-c']
    betvec = betvec0
    magbet=[]
    num_bands=(na+nb-2)

    def energiesbet(kx,ky,bet):
        hambet=HTB_2D(na,nb,kx,ky,bet,sublpert_strength)
        enerbet,wave=np.linalg.eigh(hambet) #wave[site,k]
        return enerbet

    enerbetvec=np.zeros((num_bands,len(betvec)))
    enerbetvec2=np.zeros((num_bands,len(betvec)))
    
    for i in range(len(betvec)):
        magbet.append(compute_orbital_magnetization(na,nb,a,t1,t3,hamiltoniantype,dimercells,betvec[i],sublpert_strength,Bfield,Efield))
        for j in range(num_bands):
            enerbetvec[j,i]=energiesbet(0,0,betvec[i])[j]
            enerbetvec2[j,i]=energiesbet(K[0],K[1],betvec[i])[j]


    fig, ax_left = plt.subplots(figsize=(8, 6))

    # First band structure: phi = 0
    for j in range(num_bands):
        ax_left.plot(betvec, enerbetvec[j, :], 'r--', linewidth=1.2,linestyle='--', label=f'Bands ($k_x=0$)' if j == 0 else "")

    # Second band structure: phi = 2π/3
    for j in range(num_bands):
        ax_left.plot(betvec, enerbetvec2[j, :], 'b--', linewidth=1.2,linestyle='--', label=f'Bands ($k_x=2\pi/3$)' if j == 0 else "")

    ax_left.set_xlabel(r'$\alpha$', fontsize=14)
    ax_left.set_ylabel(r'$E(\alpha)$', fontsize=14)
    ax_left.set_title(r'Energy bands and magnetization vs $\alpha$', fontsize=15)
    ax_left.tick_params(axis='y', labelcolor='black')
    ax_left.grid(True)

    # Right axis for magnetization
    ax_right = ax_left.twinx()
    ax_right.plot(betvec, magbet, '.-k', label=r'$M(\alpha)$', markersize=5)
    ax_right.set_ylabel(r'$M(\alpha)$', fontsize=14, color='k')
    ax_right.tick_params(axis='y', labelcolor='k')

    # Combine legends
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', fontsize=10)

    plt.tight_layout()
    return

def plot_OMM_BZ(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    import matplotlib.colors as mcolors

    Nk = 101
    a = 2.42  # graphene lattice constant
    R = na + nb + 1

    # === Define real-space lattice vectors for the supercell ===
    a1_real = R * a * np.array([np.sqrt(3)/2, 0.5])
    a2_real = R * a * np.array([-np.sqrt(3)/2, 0.5])

    # === Get reciprocal lattice vectors ===
    b1, b2 = get_reciprocal_lattice(a1_real, a2_real)

    # === Build k-point grid in fractional (b1, b2) coordinates ===
    i_vals = np.linspace(-0.5, 0.5, Nk, endpoint=False)
    j_vals = np.linspace(-0.5, 0.5, Nk, endpoint=False)
    I, J = np.meshgrid(i_vals, j_vals, indexing='ij')

    kx_vals = I * b1[0] + J * b2[0]
    ky_vals = I * b1[1] + J * b2[1]

    # === Compute orbital magnetic moment over the grid ===
    delta_k = 1e-4
    num_bands = 2
    m_orb = np.zeros((num_bands, Nk, Nk))

    for i in range(Nk):
        for j in range(Nk):
            kx_scaled = kx_vals[i, j] / a1_real[0] * R * a
            ky_scaled = ky_vals[i, j] / a1_real[1] * R * a
            for band in range(num_bands):
                m_orb[band, i, j] = np.real_if_close(
                    compute_mz_antisym(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield, kx_scaled, ky_scaled, delta_k, band),
                    tol=1e8
                )

    total_magnetization = np.sum(m_orb, axis=0)
    
    # === Define red-half colormap ===
    seismic = plt.get_cmap('seismic')
    if np.min(total_magnetization)<0:
        red_half = mcolors.LinearSegmentedColormap.from_list(
        'red_half', seismic(np.linspace(0, 0.5, 256))
    )
    else:
        red_half = mcolors.LinearSegmentedColormap.from_list(
        'red_half', seismic(np.linspace(0.5, 1, 256))
    )  
    
    # === Plot ===
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        kx_vals * a1_real[0] / R,
        ky_vals * a1_real[1] / R,
        total_magnetization,
        shading='auto',
        cmap=red_half
    )
    plt.colorbar(label=r'Total $m^z(\mathbf{k})$')
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.title('Total Orbital Magnetization in BZ')

    # === Plot BZ boundary ===
    bz_vertices = []
    for n in range(6):
        vertex = (b1 + b2) / 3 * np.exp(1j * np.pi / 3 * n)  # rotate by 60 degrees steps
        bz_vertices.append([vertex.real, vertex.imag])
    bz_vertices = np.array(bz_vertices + [bz_vertices[0]])  # close loop
    
    plt.plot(bz_vertices[:, 0], bz_vertices[:, 1], 'k-', lw=0.7)

    plt.tight_layout()
    plt.show()

    return

