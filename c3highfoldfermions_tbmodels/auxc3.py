import numpy as np
import scipy.linalg as spy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import exp
import sympy as sym
from sympy import *
from IPython.display import display

def triangulene(nrows):
    ''' generates trianguelenes with zigzag edges
        number of zero modes = nrows -1
    '''
    a=2.42
    a1=(0.5*a,0.5*a*np.sqrt(3.0))
    a2=(-0.5*a,0.5*a*np.sqrt(3.0))
    #a1=(0.5*a*np.sqrt(3.0),0.5*a)
    #a2=(0.5*a*np.sqrt(3.0),-0.5*a)
    

    t=(0.5*a,0.5*a/np.sqrt(3.0))
#    print('a1=',a1)
#    print('a2=',a2)
#    print('t=',t)
    natom_row=nrows

    xi=0.0
    yi=0.0
    r0=[]
    for i in range(1,nrows+1):
        x=xi
        y=yi+t[1]-a1[1]
        r0.append((x,y,0))
        for j in range(1,natom_row+1):
            x=xi+(j-1)*a
            y=yi
            r0.append((x,y,0))
            r0.append((x+t[0],y+t[1],0.))

        x=x+a
        y=y
        r0.append((x,y,0))
        y=y-2.0*t[1]
        r0.append((x,y,0))
        natom_row=natom_row-1
        xi=xi+a1[0]
        yi=yi-a1[1]
        

    x=x-t[0]
    y=y-t[1]
    r0.append((x,y,0))
#    print(' number of atoms=',len(r0))     
    return r0

########
def sublattice(atoms):
    '''
    for a given sert of atoms,  this function returns
    an ordered list with [sign],  sign=+1 for A lattice, -1, for B sublattice

    Method:
    Computes Eigenstate with maximal and minimal energy.
    These must be electron-hole symmetruc partners
    Computes SL modes 

    '''

    param=(1,0,0,2.42)
    mat=hfirst(atoms,param)
    ener,wave=spy.eigh(mat)

    nsites=len(atoms)
    n=nsites-1
    amode=wave[:,nsites-1]+wave[:,0]
    bmode=wave[:,nsites-1]-wave[:,0]
    
    sublat=[]

    for k in np.arange(nsites):
        if abs(amode[k])>abs(bmode[k]):
            sublat.append(+1)
        else:
            sublat.append(-1)

    suma=0
    for k in sublat:
        suma=suma+k

#    print('sublattice imbalance=',suma)
    
    return sublat
    

    return
    


def fusedhyb(nrows,nrows2):
        ''' this function generates 2 face to face trianguelens
generated with trianguelene funciotion
        '''
        aCC=2.42/np.sqrt(3.)
        list1=triangulene(nrows) #1st triangulene
        listA2=triangulene(nrows2) #second triangulene
        nsites1=len(list1)
        nsites2=len(listA2)
        nsites=nsites1+nsites2


        xref1=list1[nsites1-1][0]
        xref2=listA2[nsites2-1][0]
        
        ymin1=list1[nsites1-1][1]
        ymin2=listA2[nsites2-1][1]

#       We replicate the triangle of listA2 :
#       shit all y to y-ymin+aCC/2 Therefore, the lowest tip at y=aCC/2
#       reversing y to -y
        list2=[] #output total structure

        for k in range(nsites1):
                x,y,z=list1[k]
                yp=y-ymin1+0.5*aCC
                list2.append((x-xref1,-yp,z))
        for k in range(nsites2):
                x,y,z=listA2[k]
                yp=y-ymin2+0.5*aCC
                list2.append((x-xref2,yp,z))                
        
        return list2
    
def thirdneigh(listofatoms,i,a):

    ''' for a given site i, finds creates the list of second neighbours
        a:  honeycomb lattice parameter
        a happens to be the second neighbour distance
        tol=0.1 AA
    '''

    tol=0.2
    number=0 # sets number of 1st neighbours
    neigh3=[]
    natoms=len(listofatoms)
    j=0
    acc=a/np.sqrt(3)
    a3=2.*acc
    while j<natoms and number<=4: #4 second neighbours
        if j==i:
            j=j+1
        else:
            dist=distij(listofatoms,i,j)
            if a3-tol<dist<=a3+tol:
                number=number+1
                neigh3.append(j)
            j=j+1
        
    return neigh3

def distij(listofatoms,i,j):
    x1,y1,z1=listofatoms[i]
    x2,y2,z2=listofatoms[j]
    return np.sqrt((x1-x2)**2.+(y1-y2)**2.+(z1-z2)**2.)

def firstneigh2(listofatoms,i,a):
    ''' for a given site i, finds up to 3 first neighbours and returns labels
    '''

    cutoff=a/np.sqrt(3.)
    tol=0.15
    cutoff=cutoff+tol
    
    number=0 # sets number of 1st neighbours
    neigh=[]
    natoms=len(listofatoms)
    j=0
    
    while j<natoms and number<=3:
        if j==i:
            j=j+1
        else:
            dist=distij(listofatoms,i,j)
            if dist<=cutoff:
                number=number+1
                neigh.append(j)
            j=j+1
        
    return neigh
def secondneigh(listofatoms,i,a):

    ''' for a given site i, finds creates the list of second neighbours
        a:  honeycomb lattice parameter
        a happens to be the second neighbour distance
        tol=0.1 AA
    '''

    tol=0.1
    number=0 # sets number of 1st neighbours
    neigh2=[]
    natoms=len(listofatoms)
    j=0
    
    while j<natoms and number<=6: #6 second neighbours
        if j==i:
            j=j+1
        else:
            dist=distij(listofatoms,i,j)
            if a-tol<dist<=a+tol:
                number=number+1
                neigh2.append(j)
            j=j+1
        
    return neigh2

def thirdneigh(listofatoms,i,a):

    ''' for a given site i, finds creates the list of second neighbours
        a:  honeycomb lattice parameter
        a happens to be the second neighbour distance
        tol=0.1 AA
    '''

    tol=0.2
    number=0 # sets number of 1st neighbours
    neigh3=[]
    natoms=len(listofatoms)
    j=0
    acc=a/np.sqrt(3)
    a3=2.*acc
    while j<natoms and number<=4: #4 second neighbours
        if j==i:
            j=j+1
        else:
            dist=distij(listofatoms,i,j)
            if a3-tol<dist<=a3+tol:
                number=number+1
                neigh3.append(j)
            j=j+1
        
    return neigh3

def hfirst(atoms,param):
    ''' 1st neigh hamil for 0D nanographene
        This function is used by sublattice

    '''
    nsites=len(atoms)
    t=param[0]
    t2=param[1]
    t3=param[2]
    a=param[3]

    hamil0=np.array([[0.0j for i in range(nsites)] for j in range(nsites)])
    for i in range(nsites):
        x0,y0=atoms[i][0],atoms[i][1]
        first=firstneigh2(atoms,i,a)
        number1=len(first)
        for j in range(number1):
            hamil0[i,first[j]]=t

    return hamil0

def hamil0(atoms,param):
    ''' input: list of atoms
        param[0]=t first neighbour hopping
        param[1]=t2 second neighbour
        param[2]=t3 third neighbour
        paam[3]=a
        param[4]=t_Haldane
        param[5],param[6], sublattice potential
        
        t: 1st neighb hopping (complex)
        cutoff: cutoff distance
        
         output: 1st neighbour Hamil matrix 
    '''
    nsites=len(atoms)
    t=param[0]
    t2=param[1]
    t3=param[2]
    a=param[3]
    if len(param)>4:
        tH=param[4]
    else:
        tH=0

    if len(param)>5:
        VA=param[5]
        VB=param[6]
    else:
        VA=0
        VB=0
            
    sub=sublattice(atoms) # this function uses sublattice that uses hfirst to be computed 

    
    hamil0=np.array([[0.0j for i in range(nsites)] for j in range(nsites)])
    for i in range(nsites):
        x0,y0=atoms[i][0],atoms[i][1]
        first=firstneigh2(atoms,i,a)
        second=secondneigh(atoms,i,a)
        third=thirdneigh(atoms,i,a)
        number1=len(first)
        if sub[i]>0:
            hamil0[i,i]=VA
        else:
            hamil0[i,i]=VB
            
        for j in range(number1):
            hamil0[i,first[j]]=t
            
        number2=len(second)
        for j in range(number2):
            ssj=second[j]
            x1,y1=atoms[ssj][0],atoms[ssj][1]
            xx=(x1-x0)/a
            yy=(y1-y0)/a
            theta=np.arctan2(yy,xx)
            theta=theta+24*np.pi
            
#            print(i,ssj, "theta, sin theta",theta,int(3*theta/np.pi)%2)
            fac=int(3*theta/np.pi)%2
            fac2=-1+2*fac
                       
            hamil0[i,second[j]]=t2+1j*sub[i]*tH*fac2


        number3=len(third)
        for j in range(number3):
            hamil0[i,third[j]]=t3
            
    return hamil0


