import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from engine import Engine

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'

m=Engine(n=1000,k=32)
Jofb=PchipInterpolator(m.calibration[:,0],m.calibration[:,1])

def smooth_b(t,tau):
    u=np.clip(t/tau,0.0,1.0)
    z=10*u**3-15*u**4+6*u**5
    return m.bc+(m.bo-m.bc)*z

def h2(J,eps):
    return np.array([[eps/2,-J],[-J,-eps/2]],dtype=complex)

def eig(H):
    vals,vecs=np.linalg.eigh(H)
    return vals,vecs

def transition_prob(tau,eps):
    Hc=h2(m.jc,eps); Ho=h2(m.jo,eps)
    ec,vc=eig(Hc); eo,vo=eig(Ho)
    # propagate closed ground state; in 2D unitarity fixes the common excitation probability
    psi0=vc[:,0]
    if tau==0:
        psi=psi0
    else:
        def rhs(t,psi):
            J=float(Jofb(smooth_b(t,tau)))
            return -1j*h2(J,eps)@psi
        sol=solve_ivp(rhs,(0,tau),psi0,method='DOP853',rtol=1e-10,atol=1e-12)
        psi=sol.y[:,-1]
    P=abs(np.vdot(vo[:,1],psi))**2
    return float(P), ec, eo

def tls_work(tau,eps,tc=.01,th=.08):
    P,ec,eo=transition_prob(tau,eps)
    Omc=ec[1]-ec[0]; Omo=eo[1]-eo[0]
    sc=np.tanh(Omc/(2*tc)); sh=np.tanh(Omo/(2*th))
    Wad=.5*(Omo-Omc)*(sc-sh)
    L=Omo*sc+Omc*sh
    W=Wad-P*L
    return dict(P=P,W_tls=W,Wad_tls=Wad,L_tls=L,Wnorm_tls=W/Wad if Wad!=0 else np.nan)

bias=pd.read_csv(DATA/'bias.csv')
bias['tau']=bias['tau'].astype(float)
bias['eps_over_Jo']=bias['eps_over_Jo'].astype(float)
rows=[]
for er in sorted(bias.eps_over_Jo.unique()):
    eps=er*m.jo
    # continuous TLS curve at useful log-spaced durations plus exact spatial samples
    taus=np.unique(np.r_[np.geomspace(.5,120,140),bias.loc[bias.eps_over_Jo==er,'tau'].values])
    for tau in taus:
        rows.append(dict(eps_over_Jo=er,eps=eps,tau=tau,**tls_work(float(tau),eps)))
pd.DataFrame(rows).to_csv(DATA/'bias_tls.csv',index=False)

# join at spatial sampled durations and compute mismatch diagnostics
joined=bias.merge(pd.DataFrame(rows),on=['eps_over_Jo','tau'],how='left')
joined['Wnorm_spatial']=joined.W/joined.W_ad
joined['delta_Wnorm']=joined.Wnorm_spatial-joined.Wnorm_tls
joined['loss_spatial']=joined.W_ad-joined.W
joined['loss_tls_on_spatial_budget']=joined.W_ad-joined.W_tls
joined.to_csv(DATA/'bias_tls_spatial_comparison.csv',index=False)
print(joined[['eps_over_Jo','tau','Wnorm_spatial','Wnorm_tls','P','delta_Wnorm']].to_string(index=False))

# Exact decomposition of full-spatial nonadiabatic loss into transitions within the
# lowest pair and leakage into m>=2, for selected representative points. Thermal
# population above n=1 is retained as a remainder rather than silently discarded.
def decompose_spatial(er,tau):
    a=m if er==0 else Engine(n=1000,k=32,eps=er*m.jo)
    B,D=a.propagate(tau,'smooth_b',columns=6)
    pc=np.exp(-(a.ec-a.ec.min())/.01); pc/=pc.sum()
    ph=np.exp(-(a.eo-a.eo.min())/.08); ph/=ph.sum()
    PB=abs(a.R.T@B)**2
    PD=abs(D)**2
    # excess relative to adiabatic mapping, explicitly by transitions
    def parts(P,pi,ef,c=6):
        internal=0.;leak=0.;higher_initial=0.
        for n in range(c):
            for mm in range(a.k):
                contrib=pi[n]*(ef[mm]-ef[n])*P[mm,n]
                if n<2 and mm<2:
                    internal+=contrib
                elif n<2 and mm>=2:
                    leak+=contrib
                else:
                    higher_initial+=contrib
        return internal,leak,higher_initial
    io,lo,ho=parts(PB,pc,a.eo)
    ic,lc,hc=parts(PD,ph,a.ec)
    res=a.analyze(B,D)
    return dict(eps_over_Jo=er,tau=tau,D_open=res['D_open'],D_close=res['D_close'],
                internal_open=io,leak_open_energy=lo,higher_open=ho,
                internal_close=ic,leak_close_energy=lc,higher_close=hc,
                D_total=res['D_open']+res['D_close'],
                internal_total=io+ic,leak_energy_total=lo+lc,higher_total=ho+hc,
                W=res['W'],W_ad=res['W_ad'])

parts=[]
for er in [0,.1,.2,.4,.7,1.0]:
    for tau in [10,20,40]:
        parts.append(decompose_spatial(er,tau))
pd.DataFrame(parts).to_csv(DATA/'bias_loss_decomposition.csv',index=False)
print('\nLOSS DECOMPOSITION')
print(pd.DataFrame(parts)[['eps_over_Jo','tau','D_total','internal_total','leak_energy_total','higher_total','W','W_ad']].to_string(index=False))
