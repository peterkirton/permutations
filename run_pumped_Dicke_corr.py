from pylab import figure, plot, show, contourf
from numpy import sqrt, array, linspace
from time import time

from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
from basis import setup_basis, setup_rho
from models import setup_pumped_Dicke
from propagate import time_evolve, steady, spectrum, corr_func
from expect import expect_comp, setup_convert_rho, wigner_comp
from indices import list_equivalent_elements

import sys



#system size
ntls = 2
nphot = 5

#define parameters
omega=1.0
omega0=0.5
U=0.0
g=0.1
g=g/sqrt(ntls)


gam_phi = 0.0
gam_dn = 0.2

kappa = 1.0

gp = 0.1/sqrt(ntls)
gam_up=0.8


nphot0 = 0
tmax = 20
dt = 0.1
nproc = None

# setup basis with ntls spin, each of Hilbert space dimentsion 
# 2 and photon with dimension nphot
setup_basis(ntls, 2, nphot)
print("basis setup")

#run other setup routines
list_equivalent_elements()
setup_convert_rho()
from basis import nspins, ldim_p, ldim_s

#setup inital state and calculate Liouvillian
L = setup_pumped_Dicke(omega, omega0, U, g, gp, kappa, gam_phi, gam_dn, gam_up) 

initial = setup_rho(basis(ldim_p, nphot0), basis(ldim_s,1))
print("setup L")

#operators to calculate expectation values for
na = tensor(create(ldim_p)*destroy(ldim_p), qeye(ldim_s))
sz = tensor(qeye(ldim_p), sigmaz())
a = tensor(destroy(ldim_p), qeye(ldim_s))
adag = tensor(create(ldim_p), qeye(ldim_s))


#calculate steady state
rho_ss = steady(L, initial, tol=1e-9)

tlist = linspace(0, 200, 1000)

tend = tlist[-1]
dt = tlist[1]-tlist[0]

#calculate two time correlation function
corr = corr_func(L, rho_ss, adag/ntls, a, tend+dt, dt, nproc)

#calculate steady state spectrum
spec, omlist = spectrum(L, rho_ss, adag/float(ntls), a, tlist, nproc)


figure(1)
plot(tlist, corr)
show(block=False)

figure(2)
plot(omlist, spec)
show(block=False)


