from .auxc3 import *

import numpy as np
import scipy.linalg as spy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import exp
import sympy as sym
from sympy import *
from IPython.display import display


def corners0(na,nb):
    a=1.42*np.sqrt(3)
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12

    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    larac=np.linalg.norm(Ra*a1v + Ra*a2v)
    xtest=np.zeros(lengthh)
    ytest=np.zeros(lengthh)
    a=1.42*np.sqrt(3)
    a1=(0.5*a,0.5*a*np.sqrt(3.0))
    a2=(-0.5*a,0.5*a*np.sqrt(3.0))
    for i in range(lengthh):
            xtest[i]=fusedhyb(na,nb)[i][0]
            ytest[i]=fusedhyb(na,nb)[i][1]
    xatest=xtest[0:nupt]
    yatest=ytest[0:nupt]
    xbtest=xtest[nupt:]
    ybtest=ytest[nupt:]

    indxmina=np.argwhere(xatest==min(xatest))
    indxmaxa=np.argwhere(xatest==max(xatest))

    indymina=np.argwhere(yatest==min(yatest))
    w1=(np.argwhere(xatest==min(xatest[indymina])))
    for i in range(len(w1)):
        for j in range(len(indymina)):
            if w1[i]==indymina[j]:
                w2=j
    w1=(np.argwhere(xatest==max(xatest[indymina])))
    for i in range(len(w1)):
        for j in range(len(indymina)):
            if w1[i]==indymina[j]:
                w3=j
    indxminb=np.argwhere(xbtest==min(xbtest))+nupt
    indxmaxb=np.argwhere(xbtest==max(xbtest))+nupt

    indymaxb=np.argwhere(ybtest==max(ybtest))
    w1=(np.argwhere(xbtest==min(xbtest[indymaxb])))
    for i in range(len(w1)):
        for j in range(len(indymaxb)):
            if w1[i]==indymaxb[j]:
                w4=j

    w1=(np.argwhere(xbtest==max(xbtest[indymaxb])))
    for i in range(len(w1)):
        for j in range(len(indymaxb)):
            if w1[i]==indymaxb[j]:
                w5=j

    cornerbl=np.concatenate((indxmina, indymina[w2]), axis=None)
    cornerbr=np.concatenate((indymina[w3],indxmaxa), axis=None)
    cornertl=np.concatenate((indxminb,indymaxb[w4]+nupt), axis=None)
    cornertr=np.concatenate((indymaxb[w5]+nupt,indxmaxb), axis=None)
    
    middled=[nupt-6,nupt-1,nupt-2]
    middleu=[lengthh-6,lengthh-1,lengthh-2]
    
    return cornerbl,cornerbr,cornertl,cornertr,middled,middleu,xtest,ytest


def corners(na,nb):
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    xtest=np.zeros(lengthh)
    ytest=np.zeros(lengthh)
    a=2.42
    a1=(0.5*a,0.5*a*np.sqrt(3.0))
    a2=(-0.5*a,0.5*a*np.sqrt(3.0))
    for i in range(lengthh):
            xtest[i]=fusedhyb(na,nb)[i][0]
            ytest[i]=fusedhyb(na,nb)[i][1]
    xatest=xtest[nupt:nupt+ndownt]
    yatest=ytest[nupt:nupt+ndownt]
    xbtest=xtest[:nupt]
    ybtest=ytest[:nupt]

    indymaxa=np.argwhere(yatest==max(yatest))
    indxmaxa=np.argwhere(xatest==max(xatest))
    indxmina=np.argwhere(xatest==min(xatest))
    
    w1=(np.argwhere(yatest==max(yatest[indymaxa])))
    for i in range(len(w1)):
        if xatest[w1[i]]<xatest[w1[i-1]]:
            w2=w1[i][0]
    w1=(np.argwhere(yatest==max(yatest[indymaxa])))
    for i in range(len(w1)):
        if xatest[w1[i]]>xatest[w1[i-1]]:
            w3=w1[i][0]
    
    indyminb=np.argwhere(ybtest==min(ybtest))
    indxminb=np.argwhere(xbtest==min(xbtest))
    indxmaxb=np.argwhere(xbtest==max(xbtest))
    

    w1=(np.argwhere(ybtest==min(ybtest[indyminb])))
    for i in range(len(w1)):
        if xbtest[w1[i]]<xbtest[w1[i-1]]:
            w4=w1[i][0]
    w1=(np.argwhere(ybtest==min(ybtest[indyminb])))
    for i in range(len(w1)):
        if xbtest[w1[i]]>xbtest[w1[i-1]]:
            w5=w1[i][0]

    cornerbl=np.concatenate((indxmina, w2), axis=None)
    cornerbr=np.concatenate((w3,indxmaxa), axis=None)
    cornertl=np.concatenate((indxminb,w4), axis=None)
    cornertr=np.concatenate((w5,indxmaxb), axis=None)
    
    middled=[ndownt-6,ndownt-1,ndownt-2]
    middleu=[nupt-6,nupt-1,nupt-2]
    
    return cornerbl,cornerbr,cornertl,cornertr,middled,middleu,xtest,ytest

def xycoord(na,nb,dimercells):
    
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    xvector=np.zeros(dimercells*lengthh)
    yvector=np.zeros(dimercells*lengthh)
    a=2.42
    a1=(0.5*a,0.5*a*np.sqrt(3.0))
    a2=(-0.5*a,0.5*a*np.sqrt(3.0))
    #a1=(0.5*a*np.sqrt(3.0),0.5*a)
    #a2=(0.5*a*np.sqrt(3.0),-0.5*a)
    
    #plt.figure(figsize=(2*nycells,2.4*nycells))
    for j in range(dimercells):
        for i in range(lengthh):
            if j==0:
                xvector[i]=fusedhyb(na,nb)[i][0]
                yvector[i]=fusedhyb(na,nb)[i][1]
            if j%2==1:
                xvector[j*lengthh+i]=xvector[(j-1)*lengthh+i]+Ra*a1[0]
                yvector[j*lengthh+i]=yvector[(j-1)*lengthh+i]+Ra*a1[1]

            if j%2==0 and j!=0:
                xvector[j*lengthh+i]=xvector[(j-1)*lengthh+i]+Ra*a2[0]
                yvector[j*lengthh+i]=yvector[(j-1)*lengthh+i]+Ra*a2[1]

    return xvector,yvector


def hribbon(na,nb,t1,t3,nycells):

    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    
    matrix=hamil0(fusedhyb(na,nb),[t1,0,t3,2.42])
    matrix3=np.zeros((2*lengthh,2*lengthh),dtype = 'complex_') ## two triangulenes
    matrix3[0:lengthh,0:lengthh]=matrix[:,:]
    matrix3[lengthh:2*lengthh,lengthh:2*lengthh]=matrix[:,:]
    
    for j in range(3):
        if j==1:
            tti=t1
        else:
            tti=t3
        matrix3[corners0(na,nb)[3][j],corners0(na,nb)[0][j]+lengthh]=tti
        matrix3[corners0(na,nb)[0][j]+lengthh,corners0(na,nb)[3][j]]=np.conjugate(tti)

    matrixribbon=np.zeros((nycells*2*lengthh,nycells*2*lengthh),dtype = 'complex_')
    for i in range(2*nycells):
        if i<nycells:
            matrixribbon[i*2*lengthh:(i+1)*2*lengthh,i*2*lengthh:(i+1)*2*lengthh]=matrix3[:,:]        

        if i%2==1 and i<2*nycells-1: #odd
            for j in range(3):
                if j==1:
                    tti=t1
                else:
                    tti=t3
                matrixribbon[corners0(na,nb)[2][j]+i*lengthh,corners0(na,nb)[1][j]+(i+1)*lengthh]=tti
                matrixribbon[corners0(na,nb)[1][j]+(i+1)*lengthh,corners0(na,nb)[2][j]+i*lengthh]=np.conjugate(tti)
            
    return matrixribbon

def hribbonk(na,nb,t1,t3,nycells,kx,lar):
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    mati=np.zeros((nycells*2*lengthh,nycells*2*lengthh),dtype = 'complex_')
    
    for i in range(2*nycells):
        if i%2==0 and i<(2*nycells-1):
            for j in range(3):
                if j==1:
                    tti=t1
                else:
                    tti=t3
                mati[corners0(na,nb)[2][j]+i*lengthh,corners0(na,nb)[1][j]+(i+1)*lengthh]=tti*np.exp(-1j*kx*lar)
                mati[corners0(na,nb)[1][j]+(i+1)*lengthh,corners0(na,nb)[2][j]+i*lengthh]=np.conjugate(tti*np.exp(-1j*kx*lar))          
            
        if i%2==1 and i<(2*nycells-2):
            for j in range(3):
                if j==1:
                    tti=t1
                else:
                    tti=t3
                mati[corners0(na,nb)[3][j]+i*lengthh,corners0(na,nb)[0][j]+(i+1)*lengthh]=tti*np.exp(1j*kx*lar)
                mati[corners0(na,nb)[0][j]+(i+1)*lengthh,corners0(na,nb)[3][j]+i*lengthh]=np.conjugate(tti*np.exp(1j*kx*lar))
               
    return mati

def hribbonh0mk(na,nb,h0m,hk):

    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    '''  h0m = hamiltonian with peierls phases
         hk = hamiltonian with k terms
    '''
    hribbonh0mk=h0m+hk 
    

    return hribbonh0mk

def hzz(na,nb,mat):
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y


    hribbonh0mkfluxzz=mat[nupt:-(ndownt+lengthh),nupt:-(ndownt+lengthh)]
    
    return hribbonh0mkfluxzz
    
def hzzxy(na,nb,xs,ys):

    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    
    xs2=xs[nupt:-(ndownt+lengthh)]
    ys2=ys[nupt:-(ndownt+lengthh)]
    
    return xs2,ys2

#####################################################################

def xycoordac(na,nb,nxcells):
    
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    xvector=np.zeros(2*nxcells*lengthh)
    yvector=np.zeros(2*nxcells*lengthh)
    a=2.42
    a1=(0.5*a,0.5*a*np.sqrt(3.0))
    a2=(-0.5*a,0.5*a*np.sqrt(3.0))
    #a1=(0.5*a*np.sqrt(3.0),0.5*a)
    #a2=(0.5*a*np.sqrt(3.0),-0.5*a)

    des=0
    for j in range(2*nxcells):
        for i in range(lengthh):
            if j==0:
                xvector[i]=fusedhyb(na,nb)[i][0]
                yvector[i]=fusedhyb(na,nb)[i][1]
            if j%2==1:
                if i<nupt:
                    xvector[j*lengthh+i]=xvector[(j-1)*lengthh+i]+Ra*a1[0]
                    yvector[j*lengthh+i]=yvector[(j-1)*lengthh+i]+Ra*a1[1]
                else:
                    xvector[j*lengthh+i]=xvector[(j-1)*lengthh+i]-Ra*a2[0]
                    yvector[j*lengthh+i]=yvector[(j-1)*lengthh+i]-Ra*a2[1]

            if j%2==0 and j!=0:
                if i==0:
                    des=des+1
                xvector[j*lengthh+i]=fusedhyb(na,nb)[i][0]+Ra*des*a1[0]-Ra*des*a2[0]
                yvector[j*lengthh+i]=fusedhyb(na,nb)[i][1]

    return xvector,yvector




def hribbonac(na,nb,t1,t3,nxcells):

    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    
    matrix1=hamil0(fusedhyb(na,nb),[t1,0,t3,2.42])[:,:]
    matrix2=matrix1.copy()
    
    middled=corners0(na,nb)[4]
    middleu=corners0(na,nb)[5]
    
    for i in range(3):
        matrix2[middled[i],middleu[i]]=0
        matrix2[middleu[i],middled[i]]=0
    
    
    matrix3=np.zeros((2*lengthh,2*lengthh),dtype = 'complex_') ## two triangulenes
    matrix3[0:lengthh,0:lengthh]=matrix1[:,:]
    matrix3[lengthh:2*lengthh,lengthh:2*lengthh]=matrix2[:,:]

    for j in range(3):
        if j==1:
            tti=t1
        else:
            tti=t3
        matrix3[corners0(na,nb)[3][j],corners0(na,nb)[0][j]+lengthh]=tti
        matrix3[corners0(na,nb)[0][j]+lengthh,corners0(na,nb)[3][j]]=np.conjugate(tti)
        
        matrix3[corners0(na,nb)[1][j],corners0(na,nb)[2][j]+lengthh]=tti
        matrix3[corners0(na,nb)[2][j]+lengthh,corners0(na,nb)[1][j]]=np.conjugate(tti)

    matrixribbonx=np.zeros((nxcells*2*lengthh,nxcells*2*lengthh),dtype = 'complex_')

    for i in range(2*nxcells):
        if i<nxcells:
            matrixribbonx[i*2*lengthh:(i+1)*2*lengthh,i*2*lengthh:(i+1)*2*lengthh]=matrix3[:,:]        

        if i>0 and i<2*nxcells-1:
            for j in range(3):
                if j==1:
                    tti=t1
                else:
                    tti=t3
                matrixribbonx[corners0(na,nb)[1][j]+i*lengthh,corners0(na,nb)[2][j]+(i+1)*lengthh]=tti
                matrixribbonx[corners0(na,nb)[2][j]+(i+1)*lengthh,corners0(na,nb)[1][j]+i*lengthh]=np.conjugate(tti)
                
                matrixribbonx[corners0(na,nb)[3][j]+i*lengthh,corners0(na,nb)[0][j]+(i+1)*lengthh]=tti
                matrixribbonx[corners0(na,nb)[0][j]+(i+1)*lengthh,corners0(na,nb)[3][j]+i*lengthh]=np.conjugate(tti)
        
    return matrixribbonx


def hribbonack(na,nb,t1,t3,nxcells,ky,lar):

    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    mati=np.zeros((nxcells*2*lengthh,nxcells*2*lengthh),dtype = 'complex_')
    
    middled=corners0(na,nb)[4]
    middleu=corners0(na,nb)[5]
    
    for p in range(2*nxcells):  
        if p%2==1:
            for j in range(3):
                if j==1:
                    tti=t1
                else:
                    tti=t3
                mati[middled[j]+p*lengthh,middleu[j]+p*lengthh]=tti*np.exp(1j*ky*lar)
                mati[middleu[j]+p*lengthh,middled[j]+p*lengthh]=np.conjugate(tti*np.exp(1j*ky*lar))
         
    return mati

def hribbonh0mk(na,nb,h0m,hk):
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    '''  h0m = hamiltonian 
         hk = hamiltonian with k terms
    '''
    hribbonh0mk=h0m+hk 
    

    return hribbonh0mk

def hac(na,nb,mat):
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    
    hac=mat[0:-lengthh,0:-lengthh]
       
    return hac
    
def hacxy(na,nb,xs,ys):

    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y
    #dimzz=2*lengthh*nycelln-nupt-(ndownt+lengthh)

    
    xs2=xs[0:-lengthh]
    ys2=ys[0:-lengthh]
    
    return xs2,ys2


def peierlsphases(mati,xs,ys,lar,phi): 
    
    '''switch xs-ys when in armchair configuration'''
    
    area = area=0.5*3*np.sqrt(3)*(1.42)**2
         
    hribbonh0mkflux=np.zeros((len(xs),len(xs)),dtype = 'complex_')
    #resultp=np.zeros((len(xs),len(xs)),dtype = 'complex_')
    #phase=(-eBhbar*(ys[j]+ys[i])*(xs[j]-xs[i]))/2
    for j in range(len(xs)):
        for i in range(len(xs)):
            if mati[i,j]!=0:
                if i<j:
                    if (xs[j]-xs[i])>lar/2:
                        phase=(-2*np.pi*phi/area)*(ys[j]+ys[i])*(xs[j]-xs[i]-lar)/2
                    elif (xs[j]-xs[i])<-lar/2:
                        phase=(-2*np.pi*phi/area)*(ys[j]+ys[i])*(xs[j]-xs[i]+lar)/2
                    else:
                        phase=(-2*np.pi*phi/area)*(ys[j]+ys[i])*(xs[j]-xs[i])/2
                if i>j:
                    if (xs[j]-xs[i])<-lar/2:
                        phase=(-2*np.pi*phi/area)*(ys[j]+ys[i])*(xs[j]-xs[i]+lar)/2
                    elif (xs[j]-xs[i])>lar/2:
                        phase=(-2*np.pi*phi/area)*(ys[j]+ys[i])*(xs[j]-xs[i]-lar)/2
                    else:
                        phase=(-2*np.pi*phi/area)*(ys[j]+ys[i])*(xs[j]-xs[i])/2
                
                hribbonh0mkflux[i,j]=mati[i,j]*np.exp(1j*phase)
                #resultp[i,j]=np.exp(1j*phase)
                
    return hribbonh0mkflux  #resultp #



def Hmultiorbital_ZZribbon(value,na,nb,nycelln,kxval,t1,t3,Bfield):
    
    
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y
    dimzz=2*lengthh*nycelln-nupt-(ndownt+lengthh)


    xs=xycoord(na,nb,nycelln)[0]
    ys=xycoord(na,nb,nycelln)[1]

    
    if value==0:
        m4=hribbon(na,nb,t1,t3,nycelln)
        m2=peierlsphases(m4,ys,xs,larzz,Bfield)
        m=hzz(na,nb,m2)
    elif value==1:
        m3=hribbonk(na,nb,t1,t3,nycelln,kxval,1)
        m2=peierlsphases(m3,ys,xs,larzz,Bfield)
        m=hzz(na,nb,m2)

    return m


def Hmultiorbital_ACribbon(value,na,nb,nxcelln,kyval,t1,t3,Bfield):
    
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larac=np.linalg.norm(Ra*a1v + Ra*a2v)    
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y
    dimzz=2*lengthh*nxcelln-lengthh


    xs=xycoordac(na,nb,nxcelln)[0]
    ys=xycoordac(na,nb,nxcelln)[1]

    
    if value==0:
        m4=hribbonac(na,nb,t1,t3,nxcelln)
        m2=peierlsphases(m4,ys,xs,larac,Bfield)
        m=hac(na,nb,m2)
    elif value==1:
        m3=hribbonack(na,nb,t1,t3,nxcelln,kyval,1)
        m2=peierlsphases(m3,ys,xs,larac,Bfield)
        m=hac(na,nb,m2)
    return m






def hamil0sep(na,nb,atoms,param):
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
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y
    
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
    for i in range(nupt+5):
        if (i%(nupt-1)==0 and i<nsites-2):
            hamil0[i,i+ndownt]=0 
            hamil0[i+ndownt,i]=0 
            
    return hamil0

def hribbonsep(na,nb,t1,t3,dimercells):
    nycels=dimercells+2
    a=2.42
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y

    matrix=hamil0sep(na,nb,fusedhyb(na,nb),[t1,0,0,a])
    #plt.plot(sorted(np.linalg.eigvals(matrix)),'.')
    #matrix3=np.zeros((lengthh,lengthh),dtype = 'complex_') ## two triangulenes
    #matrix3[0:lengthh,0:lengthh]=matrix[:,:]
    #matrix3[lengthh:2*lengthh,lengthh:2*lengthh]=matrix[:,:]

    matrixribbon=np.zeros((nycels*lengthh,nycels*lengthh),dtype = 'complex_')
    for i in range(nycels):
        matrixribbon[i*lengthh:(i+1)*lengthh,i*lengthh:(i+1)*lengthh]=matrix[:,:]  
    return matrixribbon

def peierlsphasesad(xs,ys,lar,phi,indi,indj): 
    
    '''switch xs-ys when in armchair configuration'''
         
    area = area=0.5*3*np.sqrt(3)*(1.42)**2
    #print(indj)
    #hribbonh0mkflux=np.zeros((len(xs),len(xs)),dtype = 'complex_')
    #phase=(-eBhbar*(ys[j]+ys[i])*(xs[j]-xs[i]))/2
    if indi<indj:
        if (xs[indj]-xs[indi])>lar/2:
            phase=(-2*np.pi*phi/area)*(ys[indj]+ys[indi])*(xs[indj]-xs[indi]-lar)/2
        elif (xs[indj]-xs[indi])<-lar/2:
            phase=(-2*np.pi*phi/area)*(ys[indj]+ys[indi])*(xs[indj]-xs[indi]+lar)/2
        else:
            phase=(-2*np.pi*phi/area)*(ys[indj]+ys[indi])*(xs[indj]-xs[indi])/2
    if indi>indj:
        if (xs[indj]-xs[indi])<-lar/2:
            phase=(-2*np.pi*phi/area)*(ys[indj]+ys[indi])*(xs[indj]-xs[indi]+lar)/2
        elif (xs[indj]-xs[indi])>lar/2:
            phase=(-2*np.pi*phi/area)*(ys[indj]+ys[indi])*(xs[indj]-xs[indi]-lar)/2
        else:
            phase=(-2*np.pi*phi/area)*(ys[indj]+ys[indi])*(xs[indj]-xs[indi])/2
                
    #hribbonh0mkflux[i,j]=mati[i,j]*np.exp(1j*phase)
    resultp=np.exp(1j*phase)
                
    return resultp  #resultp #


def Electricf_T(na,nb,a,nymonomer,E_field):
    taua=np.zeros((na-1,na-1),dtype='complex_')
    taub=np.zeros((nb-1,nb-1),dtype='complex_')
    
    ev=1
    taua[range(na-1), range(na-1)] = ev * E_field 
    taub[range(nb-1), range(nb-1)] = ev * E_field 
    
    matE=np.zeros(((na+nb-2)*nymonomer//2,(na+nb-2)*nymonomer//2),dtype='complex_')
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    for i in range(nymonomer):
        posy=Ra*a1v[1]*i//2
        if i%2==0:
            matE[(na+nb-2)*(i//2):(na+nb-2)*(i//2)+(nb-1),(na+nb-2)*(i//2):(na+nb-2)*(i//2)+(nb-1)]=taub*posy
        if i%2==1:
            matE[(na+nb-2)*((i-1)//2)+nb-1:(na+nb-2)*((i+1)//2),(na+nb-2)*((i-1)//2)+nb-1:(na+nb-2)*((i+1)//2)]=taua*posy
    return matE
    




def eivdimer(na,nb,m,indmonomer): #even down monomer, odd up monomer
        

    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y
    if indmonomer%2==0:
        ws4,vs4=np.linalg.eigh(m[(indmonomer//2)*lengthh:(indmonomer//2)*lengthh+ndownt,(indmonomer//2)*lengthh:(indmonomer//2)*lengthh+ndownt])
    elif indmonomer%2==1:
        ws4,vs4=np.linalg.eigh(m[((indmonomer+1)//2)*lengthh-nupt:((indmonomer+1)//2)*lengthh,((indmonomer+1)//2)*lengthh-nupt:((indmonomer+1)//2)*lengthh])
    return ws4[np.where(np.isclose([ws4],[0]))[1]],vs4[:,np.where(np.isclose([ws4],[0]))[1]]



def realcoordzz(na,nb,dimercells):
    
    nycels=dimercells+2
    #print(nycels)
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    
    xt=xycoord(na,nb,nycels)[0]
    yt=xycoord(na,nb,nycels)[1]
    xt1=hzzxy(na,nb,xt,yt)[0]
    yt1=hzzxy(na,nb,xt,yt)[1]
    #print(len(xt1)/lengthh)
    return xt1,yt1

def realcoordac(na,nb,dimercells):
    
    nycels=int((dimercells+2)//2)
    #print(nycels)
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    
    xt=xycoordac(na,nb,nycels)[0]
    yt=xycoordac(na,nb,nycels)[1]
    xt1=hacxy(na,nb,xt,yt)[0]
    yt1=hacxy(na,nb,xt,yt)[1]
    #print(len(xt1)/lengthh)
    return xt1,yt1

def auxin(na,nb,dimercells,phi):
    
    area=0.5*3*np.sqrt(3)*(1.42)**2
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12

    xt1=realcoordzz(na,nb,dimercells)[0]
    yt1=realcoordzz(na,nb,dimercells)[1]
    #print(len(xt1))
    m1=hzz(na,nb,hribbonsep(na,nb,1,0,dimercells))
    #print(shape(m1))
    m2=peierlsphases(m1,xt1,yt1,larzz,phi)
    #print(shape(m2))
    return m2,xt1,yt1


    
def taus(mataux,na,nb,indbot,last,phi,nymonomer):#all taus defined for bottom triangulene 0<indbot1<49
    m2=mataux[0]
    
    xt1=mataux[1]
    yt1=mataux[2]
    
    tau1=np.zeros((na-1,nb-1),dtype='complex_')
    tau2=np.zeros((nb-1,na-1),dtype='complex_')
    tau3=np.zeros((nb-1,na-1),dtype='complex_')
    
    link1up=[corners(na,nb)[4][0],corners(na,nb)[4][2]]#with link1+33
    link2up=[corners(na,nb)[1][2],corners(na,nb)[1][0]]#with link3+33
    link3up=[2,0]#with link2+33

    link1bot=[corners(na,nb)[5][0],corners(na,nb)[5][2]]#with link1+33
    link3bot=[corners(na,nb)[3][2],corners(na,nb)[3][0]]#with link3+33
    link2bot=[2,0]#with link2+33
      
    
    lengthh=(na-1)*(na+5)+(nb-1)*(nb+5)+12
    #print("ini matrix:",len(m2)/lengthh)
    #print("matrix:",len(m2)/lengthh)
    
    nupt=(na-1)*(na+5)+6    ## min in y 
    ndownt=(nb-1)*(nb+5)+6  ## max in y
    nycels=nymonomer+1
    area=0.5*3*np.sqrt(3)*(1.42)**2
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    #xt1=realcoordzz(na,nb,nymonomer)[0]
    #yt1=realcoordzz(na,nb,nymonomer)[1]
    #print("vector:",len(xt1)/lengthh)
    #print(indbot)
    if indbot%2==1:
        if indbot+1<last:
            for i in range(na-1):
                for j in range(nb-1):
                    #print('.')
                    #print(len(xt1),link1up[0]+lengthh*((indbot+1)//2),link1up[1]+lengthh*((indbot+1)//2),indbot)
                    #print(peierlsphasesad(xt1,yt1,larzz,phi,link1bot[0]+ndownt+lengthh*((indbot-1)//2),link1up[0]+lengthh*((indbot+1)//2)))
                    tau1[i,j]=peierlsphasesad(xt1,yt1,larzz,phi,link1bot[0]+ndownt+lengthh*((indbot-1)//2),link1up[0]+lengthh*((indbot+1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot)[1][link1bot[0],i])*eivdimer(na,nb,m2,indbot+1)[1][link1up[0],j]+peierlsphasesad(xt1,yt1,larzz,phi,link1bot[1]+ndownt+lengthh*((indbot-1)//2),link1up[1]+lengthh*((indbot+1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot)[1][link1bot[1],i])*eivdimer(na,nb,m2,indbot+1)[1][link1up[1],j]
                    #print(tau1)
        for i in range(nb-1):
            for j in range(na-1):
                if indbot in range(1,last,4):
                    #print(link3up[0]+lengthh*((indbot-1)//2),link3bot[0]+ndownt+lengthh*((indbot-1)//2))
                    #print(link2up[0]+ndownt+lengthh*((indbot-1)//2),link2bot[0]+lengthh*((indbot+1)//2))
                    tau2[i,j]=peierlsphasesad(xt1,yt1,larzz,phi,link2up[0]+lengthh*((indbot-1)//2),link2bot[0]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link2up[0],i])*eivdimer(na,nb,m2,indbot)[1][link2bot[0],j]+peierlsphasesad(xt1,yt1,larzz,phi,link2up[1]+lengthh*((indbot-1)//2),link2bot[1]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link2up[1],i])*eivdimer(na,nb,m2,indbot)[1][link2bot[1],j]
                    tau3[i,j]=peierlsphasesad(xt1,yt1,larzz,phi,link3up[0]+lengthh*((indbot-1)//2),link3bot[0]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link3up[0],i])*eivdimer(na,nb,m2,indbot)[1][link3bot[0],j]+peierlsphasesad(xt1,yt1,larzz,phi,link3up[1]+lengthh*((indbot-1)//2),link3bot[1]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link3up[1],i])*eivdimer(na,nb,m2,indbot)[1][link3bot[1],j]
                else:
                    #print(".")
                    tau2[i,j]=peierlsphasesad(xt1,yt1,larzz,phi,link3up[0]+lengthh*((indbot-1)//2),link3bot[0]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link3up[0],i])*eivdimer(na,nb,m2,indbot)[1][link3bot[0],j]+peierlsphasesad(xt1,yt1,larzz,phi,link3up[1]+lengthh*((indbot-1)//2),link3bot[1]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link3up[1],i])*eivdimer(na,nb,m2,indbot)[1][link3bot[1],j]
                    tau3[i,j]=peierlsphasesad(xt1,yt1,larzz,phi,link2up[0]+lengthh*((indbot-1)//2),link2bot[0]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link2up[0],i])*eivdimer(na,nb,m2,indbot)[1][link2bot[0],j]+peierlsphasesad(xt1,yt1,larzz,phi,link2up[1]+lengthh*((indbot-1)//2),link2bot[1]+ndownt+lengthh*((indbot-1)//2))*np.conjugate(eivdimer(na,nb,m2,indbot-1)[1][link2up[1],i])*eivdimer(na,nb,m2,indbot)[1][link2bot[1],j]    
    return tau1,tau2,tau3


def hamribbon_red(na,nb,t1,t3,mataux,nymonomer,Efield,phi):
    matrix3=np.zeros(((na+nb-2)*int(nymonomer/2),(na+nb-2)*int(nymonomer/2)),dtype = 'complex_') ## two triangulenes
    #print(shape(matrix3))
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    #print("final matrix bef loop:",len(matrix3)/(na+nb-2))   
    for i in range(nymonomer-1):
        if i%2==0:
            tau2=t3*taus(mataux,na,nb,i+1,nymonomer,phi,nymonomer)[1]
            matrix3[(na+nb-2)*(i//2):(na+nb-2)*(i//2)+nb-1,(na+nb-2)*(i//2)+nb-1:(na+nb-2)*(i//2+1)]=tau2
            matrix3[(na+nb-2)*(i//2)+nb-1:(na+nb-2)*(i//2+1),(na+nb-2)*(i//2):(na+nb-2)*(i//2)+nb-1]=np.transpose(np.conjugate(tau2))
        else:
            tau1=t3*taus(mataux,na,nb,i,nymonomer,phi,nymonomer)[0]
            matrix3[(na+nb-2)*(i-1)//2+nb-1:(na+nb-2)*(i+1)//2,(na+nb-2)*(i+1)//2:(na+nb-2)*(i+1)//2+(nb-1)]=tau1
            matrix3[(na+nb-2)*(i+1)//2:(na+nb-2)*(i+1)//2+(nb-1),(na+nb-2)*(i-1)//2+nb-1:(na+nb-2)*(i+1)//2]=np.transpose(np.conjugate(tau1))
    
    #matrix3[(na-1)*(nymonomer-1):(na-1)*(nymonomer),(nb-1)*(nymonomer-1):(nb-1)*(nymonomer)]=Eonsite(na,nb,Ra*a1v[1]*(nymonomer-1),Efield)
    #print("final matrix:",len(matrix3))       
    return matrix3

def hamribbonk_red(na,nb,t1,t3,mataux,kx,nymonomer,larzz,phi):
    matrix3=np.zeros(((na+nb-2)*int(nymonomer/2),(na+nb-2)*int(nymonomer/2)),dtype = 'complex_') ## two triangulenes
    for i in range(nymonomer-1):
        if i%2==0:
            tau3=t3*taus(mataux,na,nb,i+1,nymonomer,phi,nymonomer)[2]
            if i in range(0,nymonomer,4):
                matrix3[(na+nb-2)*(i//2):(na+nb-2)*(i//2)+nb-1,(na+nb-2)*(i//2)+nb-1:(na+nb-2)*(i//2+1)]=tau3*np.exp(1j*kx*larzz)
                matrix3[(na+nb-2)*(i//2)+nb-1:(na+nb-2)*(i//2+1),(na+nb-2)*(i//2):(na+nb-2)*(i//2)+nb-1]=np.transpose(np.conjugate(tau3)*np.exp(-1j*kx*larzz))
            else:
                matrix3[(na+nb-2)*(i//2):(na+nb-2)*(i//2)+nb-1,(na+nb-2)*(i//2)+nb-1:(na+nb-2)*(i//2+1)]=tau3*np.exp(-1j*kx*larzz)
                matrix3[(na+nb-2)*(i//2)+nb-1:(na+nb-2)*(i//2+1),(na+nb-2)*(i//2):(na+nb-2)*(i//2)+nb-1]=np.transpose(np.conjugate(tau3)*np.exp(1j*kx*larzz))            
    #print("final matrix:",len(matrix3))   
    return matrix3


def sub_pert(na,nb,alpha):
    m=np.zeros((na+nb-2,na+nb-2))
    m[range(na-1), range(na-1)] = alpha
    m[range(na-1,na+nb-2), range(na-1,na+nb-2)] = -alpha
    return m

def orb_pert(na,nb,alpha):
    m=np.zeros((na+nb-2,na+nb-2))
    subla=np.array([0,alpha,-alpha])[-na+4:]
    sublb=np.array([0,alpha,-alpha])[-nb+4:]
    if na==2:
        subla=np.array([alpha])
    if nb==2:
        sublb=np.array([alpha])
    #subla=np.linspace(alpha,-alpha,na-1)
    #sublb=np.linspace(alpha,-alpha,nb-1)
    for i in range(na-1):
        m[i,i] = subla[i]
    for i in range(na-1,na+nb-2):
        m[i,i] = sublb[i-na+1]
    return m


def orbperturbation(na,nb,alpha,nmono):
    Vorb = orb_pert(nb,na,alpha)
    Vorbbig=np.zeros([(na+nb-2)*nmono//2,(na+nb-2)*nmono//2])
    for i in range(nmono//2):
        Vorbbig[(na+nb-2)*i:(na+nb-2)*(i+1),(na+nb-2)*i:(na+nb-2)*(i+1)]=Vorb
    return Vorbbig

def sublperturbation(na,nb,alpha,nmono):
    Vsub = sub_pert(nb,na,alpha)
    Vsubbig=np.zeros([(na+nb-2)*nmono//2,(na+nb-2)*nmono//2])
    for i in range(nmono//2):
        Vsubbig[(na+nb-2)*i:(na+nb-2)*(i+1),(na+nb-2)*i:(na+nb-2)*(i+1)]=Vsub
    return Vsubbig

def Hmultiorbital_ZZribbon_red(value,na,nb,mataux,nymonomer,kxval,t1,t3,Bfield,Efield,orbpert_strength,sublpert_strength):
    
    a=2.42
    a1v=np.array([0.5*a,0.5*a*np.sqrt(3.0)])
    a2v=np.array([-0.5*a,0.5*a*np.sqrt(3.0)])
    Ra=na+nb+1
    larzz=np.linalg.norm(Ra*a2v - Ra*a1v)
    
    
    if value==0:
        mat=hamribbon_red(na,nb,t1,t3,mataux,nymonomer,Efield,Bfield)
        mat+=(orbperturbation(na,nb,orbpert_strength,nymonomer)+sublperturbation(na,nb,sublpert_strength,nymonomer))

    elif value==1:
        mat=hamribbonk_red(na,nb,t1,t3,mataux,kxval,nymonomer,1,Bfield)
    return mat
    
    
    

# Example triangulene-based tight-binding model (minimal version)

def HTB_2D1(na,nb,kx,ky,mataux,taumat):

    a = 2.42      # graphene lattice constant (meters)
    R = (na+nb+1)
    a1 = R*np.array([np.sqrt(3)/2,1/2])*a
    a2 = R*np.array([-np.sqrt(3)/2,1/2])*a
    phi1 = kx * a1[0] + ky * a1[1]
    phi2 = kx * a2[0] + ky * a2[1]
    
    tau1=t3*taumat[0]
    tau2=t3*taumat[1]
    tau3=t3*taumat[2]  
    #print(tau1,tau2,tau3)
    fmatrix=1*tau1+np.exp(1j*(phi1))*np.transpose(tau2)+np.exp(1j*(phi2))*np.transpose(tau3)
    matfin=np.zeros((na+nb-2,na+nb-2),dtype = 'complex_')
    matfin[0:na-1,na-1:na+nb-2]=fmatrix
    matfin[na-1:na+nb-2,0:na-1]=np.transpose(np.conjugate(fmatrix))
     
    return matfin

def HTB_2D(na,nb,kx,ky,orbpert_strength,sublpert_strength):
    
    pert=orb_pert(na,nb,orbpert_strength)+sub_pert(na,nb,sublpert_strength)
    
    a = 2.42#1#2.46e-10      # graphene lattice constant (meters)
    R = (na+nb+1)
    a1 = R*np.array([np.sqrt(3)/2,1/2])*a
    a2 = R*np.array([-np.sqrt(3)/2,1/2])*a
    phi1 = kx * a1[0] + ky * a1[1]
    phi2 = kx * a2[0] + ky * a2[1]

    t3 = 0.1
    if na+nb==8:
        tta =np.array([[1/np.sqrt(12),1/np.sqrt(21),1/np.sqrt(21)]])
        ttb =np.array([1/np.sqrt(12),1/np.sqrt(21),1/np.sqrt(21)])
    if na+nb==5: 
        if na==2:
            tta =np.array([[1/np.sqrt(6)]])
            ttb =np.array([1/np.sqrt(11),1/np.sqrt(11)])
        if nb==2:
            tta =np.array([[1/np.sqrt(11),1/np.sqrt(11)]])
            ttb =np.array([1/np.sqrt(6)])
    if na+nb==6:
        tta =np.array([[1/np.sqrt(11),1/np.sqrt(11)]])
        ttb =np.array([1/np.sqrt(11),1/np.sqrt(11)])
    if na+nb==4:
        tta =np.array([[1/np.sqrt(6)]])
        ttb =np.array([1/np.sqrt(6)])
    
    ttef=2*t3*tta.T*ttb
    matexp1=np.zeros((na-1,nb-1),dtype = 'complex_')
    matexp2=np.zeros((na-1,nb-1),dtype = 'complex_')
#wsa=np.array([0,np.pi/3,-np.pi/3])[-na1+4:]
#wsb=np.array([0,np.pi/3,-np.pi/3])[-nb1+4:]
#print(wsa)
    wsa=np.array([0,2*np.pi/3,-2*np.pi/3])[-na+4:]
    wsb=np.array([0,2*np.pi/3,-2*np.pi/3])[-nb+4:]
    if na==2:
        wsa=np.array([0])
    if nb==2:
        wsb=np.array([0])
        #wsa=np.linspace(-np.pi/3,np.pi/3,(na-1))
        #wsb=np.linspace(-np.pi/3,np.pi/3,(nb-1))
    #print(wsa,wsb)
    for i in range(len(wsa)):
        for j in range(len(wsb)):
            matexp1[i,j]=np.exp(1j*(wsb[j]-wsa[i]))
            matexp2[i,j]=np.exp(-1j*(wsb[j]-wsa[i]))
    #print(matexp1)
    
    fmatrix=ttef*(1+np.exp(1j*(phi1))*matexp1+np.exp(1j*(phi2))*matexp2)
    
    matfin=np.zeros((na+nb-2,na+nb-2),dtype = 'complex_')
    matfin[0:na-1,na-1:na+nb-2]=fmatrix
    matfin[na-1:na+nb-2,0:na-1]=np.transpose(np.conjugate(fmatrix))
     
    return matfin+pert

def h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,value,kx,ky=0):
    
    if hamiltoniantype=='HTB_reduced_ZZribbon':
        mataux=auxin(na,nb,dimercells,Bfield)
        m=Hmultiorbital_ZZribbon_red(value,na,nb,mataux,2*dimercells,kx,t1,t3,Bfield,Efield,orbpert_strength,sublpert_strength)
        if value==0:
            m+=Electricf_T(na,nb,a,2*dimercells,Efield)
    elif hamiltoniantype=='HTB_ZZribbon':
        m=Hmultiorbital_ZZribbon(value,na,nb,int(dimercells)+2,kx,t1,t3,Bfield)
    elif hamiltoniantype=='HTB_ACribbon':
        m=Hmultiorbital_ACribbon(value,na,nb,int((dimercells+2)//2),kx,t1,t3,Bfield)
    elif hamiltoniantype=='HTB_2D':
        m=HTB_2D(na,nb,kx,ky,orbpert_strength,sublpert_strength)

        
    ###### choose function from c3highfoldfermions_tbmodels:
    # 
    #
    #
    # [na,nb]-T full zigzag ribbon: Hmultiorbital_ZZribbon(na,nb,int(dimercells/2)+1,k,t1,t3,Bfield)
    # [na,nb]-T full armchair ribbon: Hmultiorbital_ZZribbon(na,nb,numbercells,k,t1,t3,Bfield)
    
    #m=Hmultiorbital_ZZribbon(value,na,nb,int(dimercells/2)+1,k,t1,t3,Bfield)
    
    
    return m



def hamiltonian_bs(k):
    ###### choose function from c3highfoldfermions_tbmodels:
    # 
    #
    #
    # [na,nb]-T full zigzag ribbon: Hmultiorbital_ZZribbon(na,nb,int(dimercells/2)+1,k,t1,t3,Bfield)
    # [na,nb]-T full armchair ribbon: Hmultiorbital_ZZribbon(na,nb,numbercells,k,t1,t3,Bfield)
    #ham=Hmultiorbital_ZZribbon(value,na,nb,int(dimercells/2)+1,k,t1,t3,Bfield)
    ham=h0hk(na,nb,a,dimercells,hamiltoniantype,t1,t3,orbpert_strength,sublpert_strength,Bfield,Efield,1,k)
    return ham





