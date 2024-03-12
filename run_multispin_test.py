#!/usr/bin/env python
import numpy as np
from time import time
import pickle, sys, os
from pprint import pprint

from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
from basis import setup_basis, setup_rho
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, setup_convert_rhos_from_ops, get_rdms
from indices import list_equivalent_elements
from models import setup_counterlaser

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', **{'size':14})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

#system size
ntls = 7
nphot = 4

#define parameters
lamb = 0.01
gam_c = 0.02
gam_d = 0.02
nphot0 = 0
t0 = 0.0
tf = 5 / gam_c
dt = 0.2

setup_basis(ntls, 2, nphot)
list_equivalent_elements()
setup_convert_rho()
from basis import nspins, ldim_p, ldim_s

t0=time()
L = setup_counterlaser(lamb, gam_c, gam_d, progress=True)
initial = setup_rho(basis(ldim_p,nphot0), basis(ldim_s,1))
runtime=time()-t0
print("Setup basis and L in {:.0f}s".format(runtime), flush=True)

# random selection of operators to calculate expectation values for
asp = tensor(destroy(ldim_p), sigmam())
nsz = tensor(create(ldim_p)*destroy(ldim_p), qeye(ldim_s), sigmaz())
nspsm = tensor(create(ldim_p)*destroy(ldim_p), sigmap(), sigmam())
nszszsz = tensor(create(ldim_p)*destroy(ldim_p), sigmaz(), sigmaz(), sigmaz())
szspsm = tensor(qeye(ldim_p), sigmaz(), sigmap(), sigmam())

ops = [asp,nsz,nspsm,nszszsz,szspsm]
labels = [r'\langle a \sigma^- \rangle',
          r'\langle n \sigma^z \rangle',
          r'\langle n \sigma^+ \sigma^- \rangle',
          r'\langle n \sigma^z \sigma^z \sigma^z \rangle',
          r'\langle \sigma^z \sigma^+ \sigma^- \rangle',
          ]

# create matrices to convert compressed density matrix to reduced density
# matrix with number of spins appropriate for each operator in ops 
setup_convert_rhos_from_ops(ops)

t0=time()
resultscomp = time_evolve(L, initial, tf, dt, ops, progress=True)
tf=time()-t0

print("Time evolution complete in {:.0f}s".format(tf))

t = resultscomp.t
expect = resultscomp.expect


# (Optional) Extract reduced density matrices at each time = must pass
# save_states=True to time_evolve above.
#compressed_rhos = resultscomp.rho
#two_site_rhos = get_rdms(compressed_rhos, nrs=2)

num_plots = len(resultscomp.expect)

# plot dynamics of operators in ops
fig, axes = plt.subplots(num_plots, figsize=(5,8), sharex=True)
plot_t = np.array(t) * gam_c
axes[-1].set_xlabel(r'\(\gamma_c t\)')
for i in range(num_plots):
    if i==0:
        axes[i].plot(plot_t, expect[i].imag)
        axes[i].set_ylabel(r'\rm{{Im}}\({}\)'.format(labels[i]))
    else:
        axes[i].plot(plot_t, expect[i].real)
        axes[i].set_ylabel(r'\rm{{Re}}\({}\)'.format(labels[i]))
fig.savefig('multispin_test.pdf', bbox_inches='tight')
