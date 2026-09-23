"""Independent full-grid Crank-Nicolson check and randomized bound verification."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from engine import Engine,gibbs

ROOT=Path(__file__).resolve().parent
def grid_check(tau,kind,dt,n=1000):
    m=Engine(n=n,k=32)
    ec,A=m.spectrum(m.bc,6);eo,C=m.spectrum(m.bo,6)
    B=A.astype(complex);D=C.astype(complex)
    steps=int(np.ceil(tau/dt));h=tau/steps
    off=m.off
    def Haction(X,b):
        diag=1/m.dy**2+m.base+b*m.g
        Y=diag[:,None]*X
        Y[:-1]+=off[:,None]*X[1:];Y[1:]+=off[:,None]*X[:-1]
        return Y
    def step(X,b):
        diag=1/m.dy**2+m.base+b*m.g
        bands=np.zeros((3,n),complex)
        bands[0,1:]=.5j*h*off;bands[1]=1+.5j*h*diag;bands[2,:-1]=.5j*h*off
        rhs=X-.5j*h*Haction(X,b)
        return solve_banded((1,1),bands,rhs,check_finite=False)
    for i in range(steps):
        s=(i+.5)/steps
        B=step(B,float(m.barrier(s,kind)));D=step(D,float(m.barrier(1-s,kind)))
    pc=gibbs(ec,.01);ph=gibbs(eo,.08)
    ub=np.sum(B.conj()*Haction(B,m.bo),axis=0).real@pc
    ud=np.sum(D.conj()*Haction(D,m.bc),axis=0).real@ph
    w=pc@ec+ph@eo-ub-ud
    projected=m.cycle(tau,kind)['W']
    return dict(tau=tau,protocol=kind,dt=h,n=n,W_grid=w,W_projected=projected,difference=w-projected,
                norm_error=max(np.max(abs(np.sum(abs(B)**2,axis=0)-1)),np.max(abs(np.sum(abs(D)**2,axis=0)-1))))

def random_bound():
    rng=np.random.default_rng(210926);minimum=np.inf
    for size in range(2,15):
        for repeat in range(100):
            ei=np.cumsum(rng.uniform(.01,2,size));ef=np.cumsum(rng.uniform(.01,3,size))
            p=gibbs(ei,rng.uniform(.01,10))
            X=rng.normal(size=(size,size))+1j*rng.normal(size=(size,size))
            U,_=np.linalg.qr(X);P=abs(U)**2
            loss=ef@P@p-ef@p
            bound=sum((p[l]-p[l+1])*(ef[l+1]-ef[l])*P[l+1:,:l+1].sum() for l in range(size-1))
            minimum=min(minimum,loss-bound)
            assert loss-bound>=-1e-11
    return minimum

if __name__=='__main__':
    rows=[]
    for tau,kind in [(.8897975,'linear_J'),(12,'spectral_smooth'),(20,'smooth_b')]:
        for dt in [.01,.005]:
            r=grid_check(tau,kind,dt);rows.append(r);print(r,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'data/independent_grid.csv',index=False)
    print('Randomized 1300-case bound check minimum margin:',random_bound(),flush=True)
    rows=[]
    for n,k in [(1000,32),(1600,56)]:
        m=Engine(n=n,k=k)
        for tau in [5,12,80]:
            rows.append(dict(n=n,k=k,tau=tau,th=.18,**m.cycle(tau,'spectral_smooth',th=.18,columns=12)))
    pd.DataFrame(rows).to_csv(ROOT/'data/high_temperature_convergence.csv',index=False)
    rows=[]
    for count in [81,161,321]:
        rows.append(dict(calibration=count,**Engine(calibration=count).cycle(12,'spectral_smooth')))
    pd.DataFrame(rows).to_csv(ROOT/'data/calibration_convergence.csv',index=False)
