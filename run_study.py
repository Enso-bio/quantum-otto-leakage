import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,platform
import numpy as np
import scipy
import pandas as pd
from engine import Engine

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
def save(rows,name):pd.DataFrame(rows).to_csv(DATA/name,index=False)

def main():
    t0=time.time();m=Engine(n=1000,k=32)
    np.savetxt(DATA/'calibration.csv',m.calibration,delimiter=',',header='b,J,gap_even,gap_odd,risk,metric',comments='')
    reference={**m.cycle(300,'smooth_b'),'Jo':m.jo,'Jc':m.jc,
               'ec':m.ec[:10].tolist(),'eo':m.eo[:10].tolist(),
               'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    (DATA/'reference.json').write_text(json.dumps(reference,indent=2))
    times=np.unique(np.r_[0,np.geomspace(.2,160,37),.02/m.jo,.2/m.jo,1/m.jo])
    rows=[]
    for kind in ['linear_J','linear_b','smooth_b','spectral_smooth']:
        for tau in times:
            res=m.cycle(tau,kind)
            rows.append(dict(protocol=kind,tau=tau,q=m.jo*tau,**res))
            if res['sigma'] < -1e-7 or res['D_open']+res['D_close'] < -1e-7:raise RuntimeError('Thermodynamic check failed')
            if res['bound']>res['D_open']+res['D_close']+1e-7:raise RuntimeError('Leakage bound failed')
        save(rows,'protocols.csv');print('protocol',kind,'elapsed',round(time.time()-t0),flush=True)
    # Static spatial bias, not a phenomenological term in a two-state model.
    rows=[]
    for er in [0,.05,.1,.2,.4,.7,1.]:
        b=m if er==0 else Engine(n=1000,k=32,eps=er*m.jo)
        for tau in [5,10,20,40,80]:
            rows.append(dict(eps_over_Jo=er,tau=tau,**b.cycle(tau,'smooth_b')))
        save(rows,'bias.csv');print('bias',er,'elapsed',round(time.time()-t0),flush=True)
    # Nearby geometries. Keep initial doublet polarizations fixed for a fair diagnostic.
    rows=[]
    for alpha,sigma in [(2.2,.5),(2.6,.5),(2.4,.4),(2.4,.6)]:
        a=Engine(n=1000,k=32,alpha=alpha,sigma=sigma)
        tc=a.jc/(m.jc/.01);th=a.jo/(m.jo/.08)
        for kind in ['linear_J','smooth_b']:
            for tau in [0,5,10,20,40,80]:
                rows.append(dict(alpha=alpha,width=sigma,tc=tc,th=th,protocol=kind,tau=tau,**a.cycle(tau,kind,tc=tc,th=th,columns=10)))
        save(rows,'geometries.csv');print('geometry',alpha,sigma,'elapsed',round(time.time()-t0),flush=True)
    rows=[]
    for n,k,rtol in [(800,24,2e-9),(1000,32,2e-9),(1400,48,2e-10),(1800,60,2e-10)]:
        a=Engine(n=n,k=k)
        for tau,kind in [(.02/m.jo,'linear_J'),(10,'smooth_b'),(20,'smooth_b'),(40,'spectral_smooth')]:
            rows.append(dict(n=n,k=k,rtol=rtol,tau=tau,protocol=kind,**a.cycle(tau,kind,rtol=rtol,atol=rtol*.01)))
        save(rows,'convergence.csv');print('convergence',n,k,'elapsed',round(time.time()-t0),flush=True)
    print('DONE',round(time.time()-t0),flush=True)

if __name__=='__main__':main()
