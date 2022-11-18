#!/usr/bin/env python
import numpy as np
from time import time
import pickle, sys, os
from pprint import pprint

from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
from basis import setup_basis, setup_rho
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, setup_convert_rhos_from_ops
from indices import list_equivalent_elements
from models import setup_counterlaser

import matplotlib.pyplot as plt

#system size
ntls = 7
nphot = 4

#define parameters
lamb = 0.01
gam_c = 0.02
gam_d = 0.02
nphot0 = 0
t0 = 0.0
tf = 10 / gam_c
dt = 0.05

setup_basis(ntls, 2, nphot)

list_equivalent_elements()
setup_convert_rho()
from basis import nspins, ldim_p, ldim_s

t0=time()
L = setup_counterlaser(lamb, gam_c, gam_d)
initial = setup_rho(basis(ldim_p,nphot0), basis(ldim_s,1))
runtime=time()-t0
print("setup L and basis in {:.0f}s".format(runtime), flush=True)

# random selection of operators to calculate expectation values for
asp = tensor(destroy(ldim_p), sigmam())
nsz = tensor(create(ldim_p)*destroy(ldim_p), qeye(ldim_s), sigmaz())
nspsm = tensor(create(ldim_p)*destroy(ldim_p), sigmap(), sigmam())
nszszsz = tensor(create(ldim_p)*destroy(ldim_p), sigmaz(), sigmaz(), sigmaz())
szspsm = tensor(qeye(ldim_p), sigmaz(), sigmap(), sigmam())

ops = [asp,nsz,nspsm,nszszsz,szspsm]
labels = ['<as->', '<nsz>', '<ns+s->', '<nszszsz>', '<szs+s->']

# create matrices to convert compressed density matrix to reduced density
# matrix with number of spins appropriate for each operator in ops 
setup_convert_rhos_from_ops(ops)

t0=time()
resultscomp = time_evolve(L, initial, tf, dt, ops)
tf=time()-t0

print("Time evolution complete in {:.0f}s".format(tf))

t = resultscomp.t
expect = resultscomp.expect

num_plots = len(resultscomp.expect)

# plot dynamics of operators in ops
fig, axes = plt.subplots(num_plots, figsize=(5,8), sharex=True)
for i in range(num_plots):
    if i==0:
        axes[i].plot(t, expect[i].imag)
        axes[i].set_ylabel('Im{}'.format(labels[i]))
    else:
        axes[i].plot(t, expect[i].real)
        axes[i].set_ylabel('Re{}'.format(labels[i]))
axes[0].legend()
fig.savefig('out.pdf', bbox_inches='tight')
