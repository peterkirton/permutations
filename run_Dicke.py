from pylab import figure, plot, show, contourf
from numpy import sqrt, array, linspace
from time import time

from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
from basis import setup_basis, setup_rho
from models import setup_Dicke
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, wigner_comp
from indices import list_equivalent_elements


#system size
ntls = 8
nphot = 10

#define parameters
omega=1.0
omega0=1.0
U=0.0
g=0.9
g=g/sqrt(ntls)
gp=g

gam_phi = 0.1
gam_dn = 0.2
kappa = 1.0

nphot0 = 0
tmax = 20
dt = 0.1

# setup basis with ntls spin, each of Hilbert space dimentsion 
# 2 and photon with dimension nphot
setup_basis(ntls, 2, nphot)

#run other setup routines
list_equivalent_elements()
setup_convert_rho()
from basis import nspins, ldim_p, ldim_s

#setup inital state and calculate Liouvillian
L = setup_Dicke(omega, omega0, U, g, gp, kappa, gam_phi, gam_dn, 3) 
initial = setup_rho(basis(ldim_p, nphot0), basis(ldim_s,1))
print("setup L")

#operators to calculate expectation values for
na = tensor(create(ldim_p)*destroy(ldim_p), qeye(ldim_s))
sz = tensor(qeye(ldim_p), sigmaz())

#propagate
t0=time()
resultscomp = time_evolve(L, initial, tmax, dt, [na, sz])
tf=time()-t0

print("Time evollution complete")
print(tf)

#plot time evolution
figure()
plot(resultscomp.t, resultscomp.expect[0])
plot(resultscomp.t, resultscomp.expect[1])

#calculate steady state
#rho_ss = steady(L, initial, tol=1e-9)

#xvec = linspace(-5,5,100)
#w = wigner_comp(rho_ss, xvec, xvec)

#figure()
#contourf(xvec,xvec,w,100)

show(block=False)
