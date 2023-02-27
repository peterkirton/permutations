#!/usr/bin/env python
from numpy import sqrt, array, linspace, printoptions, save, real
from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
from basis import setup_basis, setup_rho
from models import setup_laser
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, setup_convert_rho_nrs, get_rdms, setup_convert_rhos_from_ops
from indices import list_equivalent_elements

class Pauli:
    p = np.array([[0,1],[0,0]], dtype=complex)
    m = np.array([[0,0],[1,0]], dtype=complex)
    x = p + m
    y = 1j * (m - p)
    z = np.array([[1,0],[0,-1]], dtype=complex)
    i = np.eye(2, dtype=complex)

#system size
ntls = 10
nphot = 2

# Need at least 3 spins as we want to inspect rdm involving up to three spins
assert ntls >= 3

gn = 0.03
g = gn/sqrt(ntls) 
kappa = 0.01
delta = 0.01
gam_tot = 0.02
gam_up = 0.015
gam_down = 0.005
gam_phi = 0.0
nphot0 = 0
tol = None

# setup basis with ntls spins, each of Hilbert space dimension 
# 2 and photon with dimension nphot
setup_basis(ntls, 2, nphot)

#run other setup routines
list_equivalent_elements()
setup_convert_rho()
from basis import nspins, ldim_p, ldim_s

t0=time()
L = setup_laser(g, delta, kappa, gam_down, gam_up, gam_phi, None, True, True) 
initial = setup_rho(basis(ldim_p, nphot0), basis(ldim_s,1))

print('setup L and basis in {:.1f}s'.format(time()-t0), flush=True)

for i in range(0,5):
    setup_convert_rho_nrs(i)

# Solve for ss
t0=time()
rho_ss = steady(L, initial, tol=tol)
runtime=time()-t0

print("Solved for steady-state in {:.0f}s".format(runtime), flush=True)

compressed_rho_list = [rho_ss]
t0=time()
rhoi = get_rdms(compressed_rho_list, nrs=1, photon=False)
rho0 = get_rdms(compressed_rho_list, nrs=0, photon=True)
rho0i = get_rdms(compressed_rho_list, nrs=1, photon=True)
rhoij = get_rdms(compressed_rho_list, nrs=2, photon=False)
rho0ij = get_rdms(compressed_rho_list, nrs=2, photon=True)
rhoijk = get_rdms(compressed_rho_list, nrs=3, photon=False)
rho0ijk = get_rdms(compressed_rho_list, nrs=3, photon=True)
rhoijkl = get_rdms(compressed_rho_list, nrs=4, photon=False)
print('Reduced density matrices retrieved in {:.0f}s'.format(time()-t0), flush=True)
nf = np.matmul(rho0[0], np.matmul(Pauli.m, Pauli.p)).trace().real
zf = np.matmul(rhoi[0], Pauli.z).trace().real
rdms = {
            'rho0':rho0,
            'rhoi':rhoi,
            'rho0i':rho0i,
            'rhoij':rhoij,
            'rho0ij':rho0ij,
            'rhoijk':rhoijk,
            'rho0ijk':rho0ijk,
            'rhoijkl':rhoijkl,
            }
expects = {'n':np.array([nf]), 'sz':np.array([zf])}
print('n\n{:.8g}\nsz\n{:.8g}'.format(expects['n'][-1], expects['sz'][-1]))
