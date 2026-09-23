import os
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from engine import Engine

ROOT=Path(__file__).resolve().parent;D=ROOT/'data';F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'savefig.dpi':200,'lines.linewidth':1.8})
colors={'linear_J':'#2763A5','linear_b':'#777777','smooth_b':'#CE711F','spectral_smooth':'#008477'}
labels={'linear_J':'Linear J','linear_b':'Linear b','smooth_b':'Quintic b','spectral_smooth':'Spectral + quintic'}
def finish(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)
def panel(ax,t):ax.text(-.12,1.06,t,transform=ax.transAxes,fontweight='bold',fontsize=11)

def main():
    m=Engine(n=1000,k=32)
    c=pd.read_csv(D/'calibration.csv');p=pd.read_csv(D/'protocols.csv');q=pd.read_csv(D/'selected_points.csv');r=json.loads((D/'reference.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(8.2,2.65),layout='constrained',gridspec_kw={'width_ratios':[1.05,1.15,.8]})
    y=np.linspace(-4,4,800)
    for b in [.5,2.5]:axs[0].plot(y,(y*y-2.4**2)**2/(8*2.4**2)+b*np.exp(-y*y/.5),label=f'b = {b}')
    axs[0].set(xlabel=r'$y=x/x_0$',ylabel=r'$V/(\hbar\omega_0)$',ylim=(0,3.5));axs[0].legend(frameon=False)
    axs[1].semilogy(c.b,2*c.J,label='Tunnel splitting',color=colors['linear_J'])
    axs[1].semilogy(c.b,c.gap_even,label=r'$E_2-E_0$ (even)',color=colors['spectral_smooth'])
    axs[1].semilogy(c.b,c.gap_odd,label=r'$E_3-E_1$ (odd)',color=colors['smooth_b'])
    axs[1].set(xlabel='Barrier amplitude b',ylabel=r'Energy / $\hbar\omega_0$');axs[1].legend(frameon=False,loc='center right',fontsize=7)
    # parity-selection schematic (not to scale)
    ax=axs[2]
    x_even,x_odd=.28,.72
    e0,e1,e2,e3=0.10,0.24,0.92,1.13
    for x,e,label in [(x_even,e0,'0 (+)'),(x_odd,e1,'1 (-)'),(x_even,e2,'2 (+)'),(x_odd,e3,'3 (-)')]:
        ax.hlines(e,x-.14,x+.14,lw=2)
        ax.text(x+.17,e,label,va='center',fontsize=7)
    ax.annotate('',xy=(x_even,e2-.04),xytext=(x_even,e0+.04),arrowprops=dict(arrowstyle='->',lw=1.2))
    ax.annotate('',xy=(x_odd,e3-.04),xytext=(x_odd,e1+.04),arrowprops=dict(arrowstyle='->',lw=1.2))
    # forbidden cross-parity channel
    ax.plot([x_even+.14,x_odd-.14],[e0,e1],ls='--',lw=1,color='0.45')
    ax.text(.50,.17,'×',ha='center',va='center',fontsize=12)
    ax.text(.50,.27,'forbidden',ha='center',va='center',fontsize=6.5)
    ax.text(.5,.95,'same-parity leakage',transform=ax.transAxes,ha='center',fontsize=7)
    ax.set_xlim(0,1);ax.set_ylim(0,1.28);ax.set_xticks([]);ax.set_yticks([]);ax.spines[['left','bottom']].set_visible(False)
    for ax0,t in zip(axs,['a','b','c']):panel(ax0,t)
    finish(fig,'fig1_spectral_scales')

    fig,axs=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
    for kind in colors:
        a=p[(p.protocol==kind)&(p.tau>0)]
        axs[0].semilogx(a.tau,a.W/a.W_ad,label=labels[kind],color=colors[kind])
        loss=(a.D_open+a.D_close)/a.W_ad
        axs[1].loglog(a.tau,np.maximum(loss,1e-12),color=colors[kind],label=labels[kind])
        axs[1].loglog(a.tau,np.maximum(a['bound']/a.W_ad,1e-12),ls=':',color=colors[kind],lw=1.4,alpha=.9)
    axs[0].axhline(0,color='black',lw=.8);axs[0].axhline(1,color='black',lw=.8,ls='--')
    axs[0].set(xlabel=r'Stroke duration $\tau=\omega_0t_w$',ylabel=r'$W_{\rm out}/W_{\rm ad}$',ylim=(-1.1,1.1),xlim=(1,160))
    axs[0].legend(frameon=False,loc='lower right')
    axs[1].axhline(1,color='black',lw=.8,ls='--');axs[1].set(xlabel=r'Stroke duration $\tau$',ylabel=r'Energy loss / $W_{\rm ad}$',ylim=(1e-8,50),xlim=(1,160))
    axs[1].text(.04,.08,r'Solid: exact loss'+'\n'+r'Dotted: lower bound $\Lambda$',transform=axs[1].transAxes,fontsize=8)
    # highlight the quintic tau=10 point
    h=q[(q.protocol=='smooth_b') & (np.isclose(q.tau,10.0))].iloc[0]
    lam=h['bound']/h.W_ad
    axs[1].plot(h.tau,lam,'o',ms=6,mfc='white',mec=colors['smooth_b'],mew=1.5,zorder=5)
    axs[1].annotate(r'$0.283\%$ leakage'+'\n'+rf'$\Lambda={lam:.3f}$',xy=(h.tau,lam),xytext=(17,3.0),fontsize=7,arrowprops=dict(arrowstyle='->',lw=.8))
    for ax,t in zip(axs,['a','b']):panel(ax,t)
    finish(fig,'fig2_work_and_bound')

    t=pd.read_csv(D/'temperature_map.csv');fig,axs=plt.subplots(1,2,figsize=(7.1,3),layout='constrained')
    for ax,kind,letter in zip(axs,['linear_J','spectral_smooth'],['a','b']):
        z=t[t.protocol==kind];grid=z.pivot(index='ratio',columns='tau',values='W')
        budget=z.pivot(index='ratio',columns='tau',values='W_ad')
        bound=z.pivot(index='ratio',columns='tau',values='bound')
        im=ax.pcolormesh(grid.columns,grid.index,grid.values/budget.values,cmap='RdBu',vmin=-1,vmax=1,shading='auto',rasterized=True)
        ax.contour(grid.columns,grid.index,grid.values,levels=[0],colors='black',linewidths=1.2)
        ax.contour(grid.columns,grid.index,bound.values-budget.values,levels=[0],colors='#F5B841',linestyles='--',linewidths=1.5)
        ax.set(xscale='log',xlabel=r'Stroke duration $\tau$',ylabel=r'$T_h/T_c$',title=labels[kind]);panel(ax,letter)
    cb=fig.colorbar(im,ax=axs,shrink=.85,pad=.025);cb.set_label(r'$W_{\rm out}/W_{\rm ad}$ (clipped)')
    finish(fig,'fig3_temperature_map')

    fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained');s=np.linspace(0,1,501)
    for kind in colors:axs[0].plot(s,m.barrier(s,kind),color=colors[kind],label=labels[kind])
    axs[0].set(xlabel=r'Reduced time $s=t/t_w$',ylabel='Barrier amplitude b');axs[0].legend(frameon=False)
    sl=pd.read_csv(D/'slew_comparison.csv');z=sl[np.isclose(sl.vmax,.2)]
    for i,kind in enumerate(colors):
        a=z[z.protocol==kind].iloc[0]
        axs[1].bar(i,a.W/a.W_ad,color=colors[kind],width=.65)
        axs[1].text(i,max(a.W/a.W_ad,0)+.045,f'{a.tau:.1f}',ha='center',fontsize=8)
    axs[1].axhline(0,color='black',lw=.7);axs[1].axhline(1,color='black',lw=.7,ls='--')
    axs[1].set(xticks=range(4),xticklabels=['Linear J','Linear b','Quintic','Spectral'],ylabel=r'$W_{\rm out}/W_{\rm ad}$',ylim=(-.4,1.22),title=r'Equal peak speed $|\dot b|_{\max}=0.2$')
    axs[1].text(.03,.03,'Numbers above bars: stroke duration',transform=axs[1].transAxes,fontsize=7)
    for ax,t0 in zip(axs,['a','b']):panel(ax,t0)
    finish(fig,'fig4_control_resources')

    z=pd.read_csv(D/'finite_contacts.csv');z=z[z.k==24]
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    for kind in ['linear_J','spectral_smooth']:
        for tau,ls in [(12,'--'),(20,'-'),(40,':')]:
            a=z[(z.protocol==kind)&(z.tau==tau)]
            axs[0].semilogx(a.contact,1e6*a.power,color=colors[kind],ls=ls,label=f'{labels[kind]}, '+r'$\tau=$'+str(tau))
        a=z[(z.protocol==kind)&(z.tau==20)]
        axs[1].semilogx(a.contact,a.eta,color=colors[kind],marker='o',ms=3,label=labels[kind])
    axs[0].set(xlabel=r'Contact duration $t_b\omega_0$',ylabel=r'$10^6\,\mathcal{P}/(\hbar\omega_0^2)$');axs[0].legend(frameon=False,fontsize=6.7)
    axs[1].axhline(r['eta'],ls='--',color='black',lw=.8,label='Adiabatic, complete reset')
    axs[1].set(xlabel=r'Contact duration $t_b\omega_0$',ylabel=r'Efficiency at $\tau=20$',ylim=(0,.88));axs[1].legend(frameon=False,fontsize=7)
    for ax,t0 in zip(axs,['a','b']):panel(ax,t0)
    finish(fig,'fig5_finite_contacts')

    z=pd.read_csv(D/'bias.csv');g=pd.read_csv(D/'geometries.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    for tau in [10,20,40,80]:
        a=z[z.tau==tau];axs[0].plot(a.eps_over_Jo,a.W/a.W_ad,'o-',ms=3,label=r'$\tau=$'+str(tau))
    axs[0].axhline(0,color='black',lw=.8);axs[0].set(xlabel=r'Spatial bias parameter $\epsilon/J_o(0)$',ylabel=r'$W_{\rm out}/W_{\rm ad}(\epsilon)$',ylim=(-.1,1.02));axs[0].legend(frameon=False)
    for (alpha,width),a in g[g.protocol=='smooth_b'].groupby(['alpha','width']):
        a=a[a.tau>0];axs[1].semilogx(a.tau,a.W/a.W_ad,'o-',ms=3,label=f'{alpha}, {width}')
    axs[1].axhline(0,color='black',lw=.8);axs[1].set(xlabel=r'Stroke duration $\tau$',ylabel=r'$W_{\rm out}/W_{\rm ad}$',ylim=(-1.1,1.05));axs[1].legend(title=r'$\alpha,\ \sigma/x_0$',frameon=False,fontsize=7)
    axs[1].set_xticks([5,10,20,40,80], labels=['5','10','20','40','80'])
    axs[1].xaxis.set_minor_formatter(plt.NullFormatter())
    for ax,t0 in zip(axs,['a','b']):panel(ax,t0)
    finish(fig,'figS1_robustness')
    print('Six figures generated.')

if __name__=='__main__':main()
