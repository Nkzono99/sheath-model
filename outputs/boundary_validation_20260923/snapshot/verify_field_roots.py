"""Audit public Fortran roots using independent adaptive velocity/potential integrals."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

Q=1.602176634e-19
EPS=8.8541878128e-12
ME=9.1093837015e-31
MI=1.67262192369e-27
OUT=Path('validation')


def audit(row):
    p={k:float(v) for k,v in row.items() if k not in ('branch',)}
    branch=row['branch']
    vth=math.sqrt(2*Q*p['Te']/ME)
    vp=math.sqrt(2*Q*p['Tpe']/ME)
    u=p['ve']/vth
    b=math.sqrt(max(0,-p['phi_min']/p['Te']))

    def density(phi,side):
        psi=phi/p['Te']
        cutoff=math.sqrt(max(0,psi+b*b))
        low=math.sqrt(max(0,psi))
        def f(w):
            a=math.sqrt(max(0,w*w-psi))
            return math.exp(-(a-u)**2)/math.sqrt(math.pi)
        nf=quad(f,cutoff,np.inf,epsabs=2e-12,epsrel=2e-10)[0]
        nr=0.0
        if (branch=='A' and side=='upper') or branch=='C':
            nr=2*quad(f,low,cutoff,epsabs=2e-12,epsrel=2e-10)[0]
        pe_cut=math.sqrt(max(0,(phi-p['phi_min'])/p['Tpe']))
        pe_amplitude=p['npe0']*math.exp((phi-p['phi_H'])/p['Tpe'])
        # PE velocity integral performed independently, including both legs below minimum.
        pe_f=pe_amplitude*quad(lambda w:math.exp(-w*w)/math.sqrt(math.pi),
                              pe_cut,np.inf,epsabs=2e-12)[0]
        pe_r=0.0
        if (branch=='A' and side=='lower') or branch=='B':
            pe_r=2*pe_amplitude*quad(lambda w:math.exp(-w*w)/math.sqrt(math.pi),
                                    0,pe_cut,epsabs=2e-12)[0]
        ni=p['ni']/math.sqrt(1-2*Q*phi/(MI*p['vi']**2))
        return (ni-p['Ne']*(nf+nr)-pe_f-pe_r)/p['ni']

    def integral(a,z,side):
        return quad(lambda phi:density(phi,side),a,z,epsabs=1e-10,epsrel=2e-8,limit=120)[0]
    if branch=='A':
        int_field=-integral(p['phi_min'],p['phi_H'],'lower')
        upper_integral=integral(p['phi_min'],0,'upper')
    else:
        int_field=integral(p['phi_H'],0,'monotonic')
        upper_integral=0.0
    field_squared=2*Q*p['ni']/EPS*int_field
    neutral=density(0,'upper' if branch=='A' else 'monotonic')
    ge=p['Ne']*vth*quad(lambda a:a*math.exp(-(a-u)**2)/math.sqrt(math.pi),
                       b,np.inf,epsabs=1e-12)[0]
    barrier=(p['phi_H']-p['phi_min'])/p['Tpe']
    gp=p['npe0']*vp*quad(lambda w:w*math.exp(-w*w)/math.sqrt(math.pi),
                        math.sqrt(max(0,barrier)),np.inf,epsabs=1e-12)[0]
    gi=p['ni']*p['vi']
    field_error=abs(field_squared-p['E_H']**2)/max(1,p['E_H']**2)
    flux_error=max(abs(ge-p['Gamma_e'])/max(1,ge),abs(gp-p['Gamma_escape'])/max(1,gp))
    j=Q*(ge-gi-gp)
    current_error=abs(j-p['J'])/(Q*max(ge,gi,gp))
    absolute_field_error=abs(field_squared-p['E_H']**2)
    samples=[]
    # Include logarithmically spaced upstream potentials, independently of the library's grid.
    fractions=np.unique(np.r_[np.linspace(0,1,17),1-10.0**np.arange(-8,-1)])
    if branch=='A':
        for side,end in [('lower',p['phi_H']),('upper',0.0)]:
            for f in fractions:
                phi=p['phi_min']+f*(end-p['phi_min'])
                samples.append(-2*Q*p['ni']/EPS*integral(p['phi_min'],phi,side))
    else:
        for f in fractions:
            phi=p['phi_H']*(1-f)
            samples.append(2*Q*p['ni']/EPS*integral(phi,0,'monotonic'))
    assert abs(neutral)<2e-6,(p['case'],branch,'neutrality',neutral)
    assert absolute_field_error<3e-9+2e-7*p['E_H']**2,(p['case'],branch,'field',absolute_field_error)
    assert min(samples)>-3e-9,(p['case'],branch,'negative profile field',min(samples))
    assert abs(upper_integral)<2e-6,(p['case'],branch,'upper integral',upper_integral)
    assert flux_error<2e-9,(p['case'],branch,'flux',flux_error)
    assert current_error<2e-9,(p['case'],branch,'current',current_error)
    return dict(case=int(p['case']),branch=branch,E_H=p['E_H'],phi_H=p['phi_H'],Ne=p['Ne'],
                neutrality_error=abs(neutral),field_squared_error=field_error,
                absolute_field_squared_error_v2_m2=absolute_field_error,
                sampled_minimum_field_squared_v2_m2=min(samples),profile_samples=len(samples),
                upstream_integral_error=abs(upper_integral),flux_relative_error=flux_error,
                current_error=current_error)


results=[audit(r) for r in csv.DictReader((OUT/'field_roots.csv').open())]
statuses=list(csv.DictReader((OUT/'field_status.csv').open()))
assert {int(r['case']) for r in results} >= {0,1,2,3,5}, 'missing previously validated probe case'
assert all(int(r['status'])==2 for r in statuses if int(r['case'])>=10)
assert any(r['branch']=='A' and r['phi_H']<0 for r in results), 'negative A not recovered'
# At nonzero PE, changing the specified field changes fitted ambient normalization.
families={b:[r for r in results if r['branch']==b and 4<=r['case']<=6] for b in ('A','B')}
variation={b:max(r['Ne'] for r in rs)/min(r['Ne'] for r in rs)-1
           for b,rs in families.items() if len(rs)>=2}
assert variation and max(variation.values())>1e-3, 'neutrality-adjusted source variation absent'
summary=dict(passed=True,roots_checked=len(results),max_neutrality_error=max(r['neutrality_error'] for r in results),
             max_field_squared_error=max(r['field_squared_error'] for r in results),
             max_flux_relative_error=max(r['flux_relative_error'] for r in results),
             ambient_normalization_fractional_range=variation,roots=results,statuses=statuses,
             unresolved_queries=[r for r in statuses if r['status']=='3'],
             scope='public Fortran candidates vs independent quadrature and sampled E² positivity; finite sampling is not a general existence proof')
(OUT/'field_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('roots','statuses')},indent=2))
