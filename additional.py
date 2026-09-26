import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from engine import Engine

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
def main():
    m=Engine(n=1000,k=32);t0=time.time()
    # Control resource accounting, including peak barrier speed.
    s=np.linspace(0,1,10001);rows=[]
    for kind in ['linear_J','linear_b','smooth_b','spectral_smooth']:
        bv=m.barrier(s,kind);slope=float(np.max(abs(np.gradient(bv,s))))
        for vmax in [.1,.2,.3]:
            tau=slope/vmax
            rows.append(dict(protocol=kind,peak_slope=slope,vmax=vmax,tau=tau,**m.cycle(tau,kind)))
    pd.DataFrame(rows).to_csv(DATA/'slew_comparison.csv',index=False)
    print('slew comparison',round(time.time()-t0),flush=True)
    # Temperature map reuses spatial propagators; here all 12 initial columns are propagated.
    rows=[]
    for kind in ['linear_J','spectral_smooth']:
        for tau in np.geomspace(1,80,25):
            B,D=m.propagate(tau,kind,columns=12)
            for ratio in np.linspace(5.2,18,33):
                rows.append(dict(protocol=kind,tau=tau,ratio=ratio,**m.analyze(B,D,th=.01*ratio)))
        pd.DataFrame(rows).to_csv(DATA/'temperature_map.csv',index=False)
        print('temperature',kind,round(time.time()-t0),flush=True)
    # Finite-time contacts in a specified, intentionally generic CPTP reset model.
    # Compute U once per tau, then reuse by temporarily replacing the propagation method.
    rows=[]
    b=Engine(n=1000,k=24)
    for tau in [8,12,20,40,80]:
        for kind in ['linear_J','spectral_smooth']:
            U,V=b.propagate(tau,kind,full=True)
            original=b.propagate
            b.propagate=lambda *a,**kw:(U,V)
            for contact in [5,10,20,40,80,160]:
                r=b.finite_bath(tau,contact,kind)
                if r['sigma_hot']<-1e-7 or r['sigma_cold']<-1e-7:raise RuntimeError('Bath entropy violation')
                rows.append(dict(protocol=kind,k=24,**r))
            b.propagate=original
        pd.DataFrame(rows).to_csv(DATA/'finite_contacts.csv',index=False)
        print('contacts',tau,round(time.time()-t0),flush=True)
    b=Engine(n=1400,k=40)
    for kind in ['linear_J','spectral_smooth']:
        rows.append(dict(protocol=kind,k=40,**b.finite_bath(20,40,kind)))
    pd.DataFrame(rows).to_csv(DATA/'finite_contacts.csv',index=False)
    # A fixed verification point used in the paper's transition-energy discussion.
    rows=[]
    for tau in [0,.02/m.jo,5,10,12,20,30,40,80]:
        for kind in ['linear_J','linear_b','smooth_b','spectral_smooth']:
            rows.append(dict(protocol=kind,tau=tau,**m.cycle(tau,kind)))
    pd.DataFrame(rows).to_csv(DATA/'selected_points.csv',index=False)
    print('DONE',round(time.time()-t0),flush=True)

if __name__=='__main__':main()
