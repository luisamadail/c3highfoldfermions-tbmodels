from .auxc3 import *
from .htest import *

import numpy as np
import scipy.linalg as spy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import exp
import sympy as sym
from sympy import *
from IPython.display import display



def plot_2D_bandstructure(na,nb,t1,t3,dimercells,hamiltoniantype,nk,orbpert_strength,sublpert_strength,Bfield,Efield):
    """
    Plot the 2D bandstructure along M–K–Γ–K′–M for a 2D hexagonal lattice.
    High-symmetry points are defined appropriately for a [3,3] triangulene superlattice.
    """
    #pert=orb_pert(na,nb,orbpert_strength)+sub_pert(na,nb,sublpert_strength)

    # -------------------------------
    # Step 1: Real-space lattice vectors for triangulene [3,3] superlattice
    # -------------------------------
    a = 2.42  # lattice constant
    R = (na+nb+1)  # superlattice scaling (effective for triangulene tiling)
    a1 = R * a*  np.array([np.sqrt(3)/2,1/2])
    a2 = R * a* np.array([-np.sqrt(3)/2,1/2])

    # -------------------------------
    # Step 2: Reciprocal lattice vectors
    # -------------------------------
    area = a1[0]*a2[1] - a1[1]*a2[0]
    b1 = 2 * np.pi * np.array([ a2[1], -a2[0]]) / area
    b2 = 2 * np.pi * np.array([-a1[1],  a1[0]]) / area

    # -------------------------------
    # Step 3: Define high symmetry points (folded BZ for [3,3] triangulene)
    # -------------------------------
    Gamma = np.array([0.0, 0.0])
    K  = (b2 - b1)/3
    Kp = (-b2 + b1)/3
    M1  = 0.5*(b2-b1)
    #M1 = np.array([-np.pi,0])/a/3
    #M2 = np.array([np.pi,0])/a/3
    M2=-M1
    path = [M1, K, Gamma, Kp, M2]
    labels = [r"$M$", r"$K$", r"$\Gamma$", r"$K'$", r"$M$"]

    # -------------------------------
    # Step 4: Interpolate k-points
    # -------------------------------
    k_path = []
    k_node = [0]
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i+1]
        segment = np.linspace(start, end, nk, endpoint=False)
        k_path.extend(segment)
        dist = np.linalg.norm(end - start)
        k_node.append(k_node[-1] + dist)

    k_path = np.array(k_path)

    # Accumulate distances
    k_dists = [0]
    for i in range(1, len(k_path)):
        dk = np.linalg.norm(k_path[i] - k_path[i-1])
        k_dists.append(k_dists[-1] + dk)
    k_dists = np.array(k_dists)

    # -------------------------------
    # Step 5: Evaluate Hamiltonian and extract bands
    # -------------------------------
    bands = []
    for kx, ky in k_path:
        Hk = h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kx,ky)
        eigs = np.linalg.eigvalsh(Hk)
        bands.append(eigs)
    #print(np.round(H_func(hamiltoniantype,0,0,0),5))
    bands = np.array(bands).T  # shape: (n_bands, n_kpoints)

    # -------------------------------
    # Step 6: Plot band structure
    # -------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    for band in bands:
        ax.plot(k_dists, band, 'k', lw=1)

    ax.set_xticks(k_node)
    ax.set_xticklabels(labels)
    ax.set_xlim(k_dists[0], k_dists[-1]+0.01)
    #ax.set_ylim(-0.12, 0.12)
    ax.set_ylabel("Energy")
    ax.set_title("2D Band Structure")
    ax.axhline(0, ls='--', color='gray', lw=0.5)
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return




def plot_rib_bandstructure(na,nb,a,dimercells,kpoints,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,low_energy=False):
    ham0=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)
    kvec=np.linspace(-np.pi,np.pi,kpoints) #adjust points if magnetic field is turned on
    ek=np.zeros([len(kvec),len(ham0)])
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    redim=int((na+nb-2)*len(ek[0,:])/lengthh)
    print("Constructing bandstructure...")
    for j in range(len(kvec)):
        kval=kvec[j]
        m=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kval)
        mtot=m+ham0
        w=np.linalg.eigvalsh(mtot)
        ek[j,:]=sorted(np.real_if_close(w, tol=100))   
    print("Done!")
    if low_energy==False:
        fig=plt.figure(figsize=(6,7))  
        plt.plot(kvec,ek,'k',linewidth=0.2)
        #plt.ylim(-0.1,0.1)
    else:
        fig=plt.figure(figsize=(6,7))  
        plt.plot(kvec,ek[:,int(len(ek.T)/2)-redim//2:int(len(ek.T)/2)+redim//2],'k',linewidth=0.7)
    if hamiltoniantype=='HTB_reduced_ZZribbon':
        print("Total number of low-energy states:",len(ek[0,:]))
    else:
        print("Total number of dimers:",len(ek[0,:])/lengthh)
        print("Total number of low-energy states:",redim)
    plt.xlabel("$k$",fontsize=15)
    plt.ylabel("$E/t$",fontsize=15)
    plt.tight_layout()
    plt.show()
    return




def plot_rib_lattice(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield):
    
    if hamiltoniantype=='HTB_ZZribbon':
        xt1=realcoordzz(na,nb,dimercells)[0]
        yt1=realcoordzz(na,nb,dimercells)[1]
    elif hamiltoniantype=='HTB_ACribbon':
        xt1=realcoordac(na,nb,dimercells)[0]
        yt1=realcoordac(na,nb,dimercells)[1]
    
    mataux=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)+h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,0.4)
    
    link1=[]
    link2=[]
    for i in range(len(mataux)):
        for j in range(len(mataux)):
            if mataux[i,j]!=0:
                link1.append(i)
                link2.append(j)
    plt.figure(figsize=(15,15))
    for i in range(len(mataux)):
        plt.plot(xt1[i],yt1[i],'r.')

        label = "{:}".format(i)
 #       plt.annotate(label, # this is the text
 #                (xt1[i],yt1[i]), # these are the coordinates to position the label
 #                textcoords="offset points", # how to position the text
 #                xytext=(0,5), # distance from text to points (x,y)
 #                ha='center') # horizontal alignment can be left, right or center


    for i in range(len(link1)):
        if np.abs(xt1[link1[i]]-xt1[link2[i]])>(na+nb-2) or np.abs(yt1[link1[i]]-yt1[link2[i]])>(na+nb-2):
            plt.plot([xt1[link1[i]],xt1[link2[i]]],[yt1[link1[i]],yt1[link2[i]]],'r-',linewidth=0.3)
        else:
            plt.plot([xt1[link1[i]],xt1[link2[i]]],[yt1[link1[i]],yt1[link2[i]]],'k-',linewidth=0.3)
    
    plt.gca().set_aspect('equal', 'box')
    plt.show()
    return


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    import matplotlib
    c1=np.array(matplotlib.colors.to_rgb(c1))
    c2=np.array(matplotlib.colors.to_rgb(c2))
        
    return matplotlib.colors.to_hex(c1*np.sin(mix/2)**2 + c2*np.cos(mix/2)**2)


def plot_wave_lattice(na,nb,a,t1,t3,hamiltoniantype,dimercells,orbpert_strength,sublpert_strength,Bfield,Efield,kxval,index):

        
    ''' Plot a state vec in atoms . very similar to plotwave, but
    the quantum state is provided as input.

    Main difference: we plot lattice links '''
    fig1=plt.figure(figsize=(20,20))   

    import cmath
    
    wave2=[]
    if hamiltoniantype=='HTB_ZZribbon':
        xt1=realcoordzz(na,nb,dimercells)[0]
        yt1=realcoordzz(na,nb,dimercells)[1]
    elif hamiltoniantype=='HTB_ACribbon':
        xt1=realcoordac(na,nb,dimercells)[0]
        yt1=realcoordac(na,nb,dimercells)[1]

        
    ws,vs=np.linalg.eigh(h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,0,0)+h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,kxval))
    vec=vs[:,index]
    
    atoms=constructing_atoms(xt1,yt1)

    c1='#9900FF' #blue
    c2='#00E600' #green
    n=100
    
    mod0,phase0=cmath.polar(vec[2])

    colorlist=('#9900FF','#00E600')
    col=np.empty(0, dtype='complex_')
#       Creates list for wave fuctin    
    for x in vec:
        mod,phase1=cmath.polar(x)
        wave2.append(round(mod,5))
        signr=np.sign(x)
        number=int(phase1*n/(2*np.pi))
        #signr=np.sign(np.imag(x))
#       print("(signr+1)/2=",np.int((signr+1)*0.5))
        col=np.append(col,colorlist[int((signr+1)*0.5)])

        #wave2.append(round((phase1-phase0),5))



    x=[]
    y=[]
    mark=[]
    wave=200*abs(np.array(vec))
    atomsize=[]
    ss=200.
    for k in range(len(atoms)):
        x.append(atoms[k][0])
        y.append(atoms[k][1])
#        atomsize.append(ss)
#    for i, txt in enumerate(atoms):
#        plt.annotate(txt, (x[i], y[i]))

#    plt.scatter(x, y, s=atomsize, color='black', alpha=0.5)
    plt.scatter(x, y, s=wave, color=col,alpha=1,zorder=10)
    plt.gca().set_aspect('equal', 'box')
    #plt.xlim([np.min(x)-15, np.max(x)+15])
    #plt.ylim([np.min(y)-15, np.max(y)+15])

#   Now we produce the links
    a=2.42


    for i in range(len(atoms)):
        first=firstneigh2(atoms,i,a)
#        third=thirdneigh(atoms,i,a)
#        print(third)
        
        
        x0,y0,z0=atoms[i]
        for j in first:
            x1,y1,z1=atoms[j]
            plt.plot((x0,x1),(y0,y1),color='k',linewidth=0.6)
#        for j in third:
#            x1,y1,z1=lista[j]
#            plt.plot((x0,x1),(y0,y1),color='green')    
    print("E=",np.round(ws[index],7))
    
    #plt.tight_layout()
    
    
    plt.show()
    
    #fig1.savefig('tri33zzstate3.svg')
    return

def constructing_atoms(xs,ys):
    atoms=[]
    for i in range(len(xs)):
        atoms.append((xs[i],ys[i],0))
    return atoms

















