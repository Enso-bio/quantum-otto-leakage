"""Spatial barrier Otto engine. All energies in hbar*omega0, time in 1/omega0.

Finite-basis propagation retains spatial deformation. No time-dependent
two-level replacement is made. Script outputs are deterministic.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp, cumulative_trapezoid

def gibbs(e,t):
    p=np.exp(-(e-e.min())/t)
    return p/p.sum()

class Engine:
    def __init__(self,n=1000,k=32,alpha=2.4,sigma=.5,bo=.5,bc=2.5,eps=0.,calibration=161):
        self.n,self.k=n,k
        self.alpha,self.sigma,self.bo,self.bc,self.eps=alpha,sigma,bo,bc,eps
        self.y=np.linspace(-7,7,n); self.dy=self.y[1]-self.y[0]
        self.g=np.exp(-self.y**2/(2*sigma**2))
        self.base=(self.y**2-alpha**2)**2/(8*alpha**2)+eps*self.y/(2*alpha)
        self.off=np.full(n-1,-.5/self.dy**2)
        self.ec,self.vc=self.spectrum(bc,k)
        self.G=self.vc.T@(self.g[:,None]*self.vc)
        self.Hc=np.diag(self.ec)
        self.Ho=self.Hc+(bo-bc)*self.G
        self.eo,self.R=np.linalg.eigh(self.Ho)
        self.jo=(self.eo[1]-self.eo[0])/2
        self.jc=(self.ec[1]-self.ec[0])/2
        bs=np.linspace(bo,bc,calibration)
        rows=[]
        for b in bs:
            e,v=self.spectrum(b,12)
            M=v.T@(self.g[:,None]*v)
            risk=0.;metric=0.
            for a in range(2):
                others=np.arange(2,12)
                risk+=np.sum(abs(M[others,a])**2/(e[others]-e[a])**4)
                metric+=np.sum(abs(M[others,a])**2/(e[others]-e[a])**2)
            rows.append([b,(e[1]-e[0])/2,e[2]-e[0],e[3]-e[1],risk**.5,metric])
        self.calibration=np.array(rows)
        self.inverse_J=PchipInterpolator(self.calibration[::-1,1],bs[::-1])
        length=cumulative_trapezoid(self.calibration[:,4],bs,initial=0)
        self.risk_inverse=PchipInterpolator(length/length[-1],bs)

    def spectrum(self,b,k):
        return eigh_tridiagonal(1/self.dy**2+self.base+b*self.g,self.off,
                               select='i',select_range=(0,k-1))

    def barrier(self,s,kind):
        """Opening stroke; closing is exactly its time reverse."""
        s=np.clip(s,0,1)
        if kind=='linear_J':return self.inverse_J(self.jc+(self.jo-self.jc)*s)
        if kind=='linear_b':return self.bc+(self.bo-self.bc)*s
        z=10*s**3-15*s**4+6*s**5
        if kind=='smooth_b':return self.bc+(self.bo-self.bc)*z
        if kind=='spectral_smooth':return self.risk_inverse(1-z)
        raise ValueError(kind)

    def propagate(self,tau,kind='linear_J',columns=6,rtol=2e-9,atol=2e-11,full=False):
        c=self.k if full else columns
        A=np.eye(self.k,c,dtype=complex)
        C=self.R[:,:c].astype(complex)
        if tau==0:return A,C
        def rhs(t,v):
            X=v.reshape(2,self.k,c)
            b=float(self.barrier(t/tau,kind));br=float(self.barrier(1-t/tau,kind))
            d=np.empty_like(X)
            d[0]=-1j*(self.ec[:,None]*X[0]+(b-self.bc)*(self.G@X[0]))
            d[1]=-1j*(self.ec[:,None]*X[1]+(br-self.bc)*(self.G@X[1]))
            return d.ravel()
        sol=solve_ivp(rhs,(0,tau),np.stack([A,C]).ravel(),method='DOP853',rtol=rtol,atol=atol)
        if not sol.success:raise RuntimeError(sol.message)
        B,D=sol.y[:,-1].reshape(2,self.k,c)
        norm=max(np.max(abs(B.conj().T@B-np.eye(c))),np.max(abs(D.conj().T@D-np.eye(c))))
        if norm>2e-6:raise RuntimeError(f'Unitarity error {norm}')
        return B,D

    def analyze(self,B,D,tc=.01,th=.08):
        c=B.shape[1]
        pc=gibbs(self.ec,tc);ph=gibbs(self.eo,th)
        PB=abs(self.R.T@B)**2;PD=abs(D)**2
        ua=pc[:c]@self.ec[:c];ub=pc[:c]@(self.eo@PB)
        uc=ph[:c]@self.eo[:c];ud=ph[:c]@(self.ec@PD)
        qh=uc-ub;qc=ua-ud;w=qh+qc
        da=ub-pc[:c]@self.eo[:c];db=ud-ph[:c]@self.ec[:c]
        wad=(ph-pc)@(self.eo-self.ec)
        # Exact finite-temperature lower bound using the lowest cut in each parity sector.
        bound=0.
        if self.eps==0:
            for i in [0,1]:
                inds=np.arange(i+2,self.k,2)
                bound+=(pc[i]-pc[i+2])*(self.eo[i+2]-self.eo[i])*PB[inds,i].sum()
                bound+=(ph[i]-ph[i+2])*(self.ec[i+2]-self.ec[i])*PD[inds,i].sum()
        leakage_o=np.sum(pc[:2]*np.sum(PB[2:,:2],axis=0))
        leakage_c=np.sum(ph[:2]*np.sum(PD[2:,:2],axis=0))
        norm=max(np.max(abs(np.sum(PB,axis=0)-1)),np.max(abs(np.sum(PD,axis=0)-1)))
        return dict(W=w,Qh=qh,Qc=qc,eta=w/qh if w>0 and qh>0 else np.nan,
                    W_ad=wad,D_open=da,D_close=db,bound=bound,
                    leakage_open=leakage_o,leakage_close=leakage_c,
                    sigma=-qh/th-qc/tc,norm_error=norm,
                    thermal_tail=max(pc[c:].sum(),ph[c:].sum()),
                    identity_error=w-(wad-da-db))

    def cycle(self,tau,kind='linear_J',tc=.01,th=.08,**kw):
        B,D=self.propagate(tau,kind,**kw)
        return self.analyze(B,D,tc,th)

    def finite_bath(self,tau,contact,kind='smooth_b',gamma=.02,tc=.01,th=.08):
        """CPTP Gibbs-reset semigroup including free Hamiltonian precession.

        Benchmark only, with no device-specific bath interpretation.
        """
        U,V=self.propagate(tau,kind,full=True)
        V=V@self.R.T
        pc=gibbs(self.ec,tc);ph=gibbs(self.eo,th)
        cold=np.diag(pc).astype(complex);hot=(self.R*ph)@self.R.T
        r=np.exp(-gamma*contact)
        Fh=(self.R*np.exp(-1j*self.eo*contact))@self.R.T
        Fc=np.diag(np.exp(-1j*self.ec*contact))
        def hot_map(X):return r*(Fh@X@Fh.conj().T)+(1-r)*hot
        def cold_map(X):return r*(Fc@X@Fc.conj().T)+(1-r)*cold
        A=cold.copy()
        for iteration in range(20000):
            B=U@A@U.conj().T
            C=hot_map(B)
            D=V@C@V.conj().T
            new=cold_map(D)
            residual=np.max(abs(new-A))
            A=new
            if residual<2e-12:break
        else:raise RuntimeError('Limit cycle did not converge')
        B=U@A@U.conj().T;C=hot_map(B);D=V@C@V.conj().T
        qh=np.trace(self.Ho@(C-B)).real;qc=np.trace(self.Hc@(A-D)).real;w=qh+qc
        def entropy(rho):
            ev=np.maximum(np.linalg.eigvalsh(rho),0);ev=ev[ev>1e-16]
            return -np.sum(ev*np.log(ev))
        return dict(tau=tau,contact=contact,gamma=gamma,W=w,Qh=qh,Qc=qc,
                    eta=w/qh if w>0 and qh>0 else np.nan,power=w/(2*(tau+contact)),
                    sigma=-qh/th-qc/tc,sigma_hot=entropy(C)-entropy(B)-qh/th,
                    sigma_cold=entropy(A)-entropy(D)-qc/tc,residual=residual,
                    min_eigenvalue=min(np.linalg.eigvalsh(x).min() for x in [A,B,C,D]))
