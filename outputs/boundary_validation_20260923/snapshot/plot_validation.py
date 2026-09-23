"""Standalone figures for the boundary validation receipts."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path('validation')
fig,axes=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
t=2.2
cut=2.0
b=np.linspace(0,12,400)
exact=np.exp(-np.maximum(0,b-cut)/t)
fit=np.exp(-b/(t+cut))
axes[0].plot(b,exact,label='Transported distribution')
axes[0].plot(b,fit,'--',label='Moment-matched Maxwell')
axes[0].axvline(cut,color='0.6',ls=':',lw=1)
axes[0].set(xlabel='Outer barrier [eV]',ylabel='Escape / outward flux at H',
            title='Controlled example: cutoff = 2 eV')
axes[0].legend(fontsize=8)
with (out/'pe_transport_saved_spectra.csv').open() as stream:
    rows=list(csv.DictReader(stream))
for name,marker in [('moments_128','o'),('spectrum_128','s'),('spectrum_32','^')]:
    group=[r for r in rows if r['archive_case']==name]
    axes[1].plot([int(r['batch']) for r in group],
                 [100*float(r['moment_fit_relative_to_bin_escape']) for r in group],
                 marker=marker,label=name,ls='none')
axes[1].axhline(0,color='0.5',lw=1)
axes[1].set(xlabel='Saved batch',ylabel='Maxwell / bin escape - 1 [%]',
            title='Same saved input and barrier',xticks=[1,2,3])
axes[1].legend(fontsize=8)
with (out/'upstream_algebraic_root_field_integral.csv').open() as stream:
    rows=list(csv.DictReader(stream))
# Show the neighborhood that violates E^2 >= 0; the full samples remain in CSV.
rows=[r for r in rows if -float(r['lower_phi_v']) <= 0.025]
axes[2].plot([-float(r['lower_phi_v']) for r in rows],
             [float(r['independently_integrated_field_squared_hat']) for r in rows],'o-')
axes[2].axhline(0,color='0.5',lw=1)
axes[2].set(xscale='log',xlabel='-potential near upstream [V]',ylabel='Normalized E squared',
            title='Algebraic A root: near-upstream failure')
axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(out/'boundary_validation.png',dpi=180)
fig.savefig(out/'boundary_validation.pdf')
with (out/'finite_reservoir_profile.csv').open() as stream:
    profile=list(csv.DictReader(stream))
profile.sort(key=lambda r:float(r['z_m']))
fig2,ax2=plt.subplots(2,1,figsize=(6.8,5),sharex=True,constrained_layout=True)
z=[float(r['z_m']) for r in profile]
ax2[0].plot(z,[float(r['potential_v']) for r in profile])
ax2[0].set(ylabel='Potential [V]',title='Constructed finite-reservoir example (outer domain only)')
ax2[1].plot(z,[float(r['electric_field_v_m']) for r in profile])
ax2[1].set(xlabel='Distance from H [m]',ylabel='Electric field [V/m]')
for a in ax2:
    a.axhline(0,color='0.5',lw=.8)
    a.grid(alpha=.2)
fig2.savefig(out/'finite_reservoir.png',dpi=180)
fig2.savefig(out/'finite_reservoir.pdf')
