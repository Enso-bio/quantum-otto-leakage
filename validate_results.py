from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;D=ROOT/'data'
def main():
    summary={}
    for name in ['protocols','temperature_map','selected_points','geometries','bias','convergence']:
        x=pd.read_csv(D/(name+'.csv'))
        tests={
            'rows':len(x),
            'first_law_max_error':float(abs(x.W-x.Qh-x.Qc).max()),
            'work_decomposition_max_error':float(abs(x.identity_error).max()),
            'min_entropy_production':float(x.sigma.min()),
            'max_norm_error':float(x.norm_error.max()),
            'max_thermal_tail':float(x.thermal_tail.max()),
            'max_bound_minus_loss':float((x.bound-x.D_open-x.D_close).max()),
        }
        assert tests['first_law_max_error']<1e-10
        assert tests['work_decomposition_max_error']<1e-8
        assert tests['min_entropy_production']>-1e-8
        assert tests['max_norm_error']<2e-6
        assert tests['max_thermal_tail']<1e-10
        assert tests['max_bound_minus_loss']<1e-9
        summary[name]=tests
    x=pd.read_csv(D/'finite_contacts.csv')
    assert min(x.sigma_hot.min(),x.sigma_cold.min())>-1e-8
    assert x.min_eigenvalue.min()>-1e-12
    assert x.residual.max()<2e-12
    summary['finite_contacts']={'rows':len(x),'min_sigma_hot':float(x.sigma_hot.min()),
         'min_sigma_cold':float(x.sigma_cold.min()),'min_density_eigenvalue':float(x.min_eigenvalue.min()),
         'max_cycle_residual':float(x.residual.max())}
    c=pd.read_csv(D/'convergence.csv');lines=[]
    for (n,k),g in c.groupby(['n','k']):
        vals=[g[g.protocol.eq('linear_J')].W.iloc[0],
              g[g.protocol.eq('smooth_b') & g.tau.eq(20)].W.iloc[0],
              g[g.protocol.eq('spectral_smooth')].W.iloc[0]]
        lines.append(f'{n} & {k} & '+' & '.join(f'{v:.8f}' for v in vals)+r' \\')
    (ROOT/'convergence_rows.tex').write_text('\n'.join(lines),encoding='utf-8')
    summary['status']='PASS'
    (D/'validation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
