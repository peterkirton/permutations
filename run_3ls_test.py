#!/usr/bin/env python
# based of run_tavisNNphlambda.py from my_perm_py3
from numpy import sqrt, array, linspace, printoptions, save, real
from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from operators import basis, tensor, destroy, create, qeye
from basis import setup_basis, setup_rho
from models import setup_3ls
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, get_rdms, setup_convert_rho_nrs, setup_convert_rhos_from_ops
from indices import list_equivalent_elements

#system size
ntls = 2
nphot = 2

#define parameters
nu = 1.0
g = 0.2
kappa = 0.05
pump = 0.05

nphot0 = 1
tmax = 10
dt = 0.1

setup_basis(ntls, 3, nphot)

#run other setup routines
list_equivalent_elements()
setup_convert_rho()
from basis import nspins, ldim_p, ldim_s

t0=time()
L = setup_3ls(nu, g, kappa, pump, progress=True, parallel=True) 
# setup initial state nphot0 photons and spins down
initial = setup_rho(basis(ldim_p, nphot0), basis(ldim_s,1))
print('setup L and basis in {:.1f}s'.format(time()-t0), flush=True)

n = tensor(create(ldim_p)*destroy(ldim_p), qeye(ldim_s))
p = tensor(qeye(ldim_p), create(ldim_s)*destroy(ldim_s))

ops = [n,p]
setup_convert_rhos_from_ops(ops)

#propagate
t0=time()
resultscomp = time_evolve(L, initial, tmax, dt, ops, atol=1e-10, rtol=1e-10)
runtime=time()-t0

print("Time evolution complete in {:.0f}s".format(runtime), flush=True)

ts = np.array(resultscomp.t)
ns = np.array(resultscomp.expect[0])
ps = np.array(resultscomp.expect[1])

compressed_rho_list = resultscomp.rho
setup_convert_rho_nrs(nrs=2) # so we can extract rdm involving two spins
rho0ij = get_rdms(compressed_rho_list, nrs=2, photon=True)
rho0i = get_rdms(compressed_rho_list, nrs=1, photon=True)
# get population of photon from rdm
ns2 = np.array([rho.dot(n) for rho in rho0i])

# Optional: export to check against qutip result
#with open('3ls_test.pkl', 'wb') as f:
#    pickle.dump(rho0ij, f)

fig, ax = plt.subplots(1, figsize=(6,3))
ax.plot(ts, ns.real, label='n')
ax.plot(ts, ns.real, label='n-check', ls='--')
ax.plot(ts, ps.real, label='pi')
ax.legend()
fig.savefig('3ls.pdf', bbox_inches='tight')
