import os
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
D=ROOT/'data'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'savefig.dpi':200,'lines.linewidth':1.7})

def panel(ax,t): ax.text(-.12,1.06,t,transform=ax.transAxes,fontweight='bold',fontsize=11)

def main():
    spatial=pd.read_csv(D/'bias.csv')
    tls=pd.read_csv(D/'bias_tls.csv')
    dec=pd.read_csv(D/'bias_loss_decomposition.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.15,2.95),layout='constrained')
    chosen=[0.0,0.1,0.4,1.0]
    for er in chosen:
        a=tls[np.isclose(tls.eps_over_Jo,er)]
        line,=axs[0].semilogx(a.tau,a.Wnorm_tls,label=rf'$\epsilon/J_o={er:g}$')
        b=spatial[np.isclose(spatial.eps_over_Jo,er) & (spatial.tau>=10)]
        axs[0].plot(b.tau,b.W/b.W_ad,'o',ms=4,color=line.get_color())
    axs[0].axhline(0,lw=.8,color='black')
    axs[0].axhline(1,lw=.8,ls='--',color='black')
    axs[0].set(xlabel=r'Stroke duration $\tau$',ylabel=r'$W_{\rm out}/W_{\rm ad}$',xlim=(8,120),ylim=(-.42,1.08))
    axs[0].legend(frameon=False,ncol=2,loc='upper right')
    axs[0].text(.04,.06,'lines: projected doublet\nmarkers: full spatial model\n($\tau=5$ spatial points omitted)',transform=axs[0].transAxes,fontsize=7.3)
    panel(axs[0],'a')

    z=dec[np.isclose(dec.tau,20)].sort_values('eps_over_Jo')
    x=z.eps_over_Jo.values
    total=z.D_total/z.W_ad
    internal=z.internal_total/z.W_ad
    leakage=z.leak_energy_total/z.W_ad
    axs[1].plot(x,total,'o-',label='total spatial loss')
    axs[1].plot(x,internal,'s--',label='within-doublet mixing')
    axs[1].plot(x,leakage,'^--',label='higher-state leakage')
    axs[1].axhline(1,lw=.8,ls=':',color='black')
    axs[1].set(xlabel=r'Static tilt $\epsilon/J_o$',ylabel=r'Nonadiabatic loss / $W_{\rm ad}$',ylim=(-.02,1.08),title=r'$\tau=20$')
    axs[1].legend(frameon=False,loc='upper left')
    panel(axs[1],'b')
    fig.savefig(F/'fig5_bias_projection.pdf',bbox_inches='tight')
    fig.savefig(F/'fig5_bias_projection.png',bbox_inches='tight')
    plt.close(fig)

    # Supplement: neighboring geometries only, since bias is now in the main text.
    g=pd.read_csv(D/'geometries.csv')
    fig,ax=plt.subplots(figsize=(3.9,2.8),layout='constrained')
    for (alpha,width),a in g[g.protocol=='smooth_b'].groupby(['alpha','width']):
        a=a[a.tau>0]
        ax.semilogx(a.tau,a.W/a.W_ad,'o-',ms=3,label=f'{alpha}, {width}')
    ax.axhline(0,color='black',lw=.8)
    ax.set(xlabel=r'Stroke duration $\tau$',ylabel=r'$W_{\rm out}/W_{\rm ad}$',ylim=(-1.1,1.05))
    ax.legend(title=r'$\alpha,\ \sigma/x_0$',frameon=False,fontsize=7)
    ax.set_xticks([5,10,20,40,80],labels=['5','10','20','40','80'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    fig.savefig(F/'figS1_geometries.pdf',bbox_inches='tight')
    fig.savefig(F/'figS1_geometries.png',bbox_inches='tight')
    plt.close(fig)
    print('Bias comparison and geometry figures generated.')

if __name__=='__main__': main()
