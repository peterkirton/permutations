class Results:
    def __init__(self):
        self.rho= []
        self.t = []
        self.expect = []

class Progress:
    def __init__(self, total, name='', start=0):
        self.start = start
        self.step = start
        self.end = total
        self.name = name
        self.percent = int(100*start//total)

    def update(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
        self.percent = int(100*self.step/self.end)
        if self.step == self.start+1:
            print('{}{:4d}%'.format(self.name, self.percent), end='', flush=True)
        else:
            print('\b\b\b\b\b{:4d}%'.format(self.percent), end='', flush=True)
        if self.step == self.end:
            print('', flush=True)

def time_evolve(L, initial, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5, progress=False):
    """time evolve matrix L from initial condition initial with step dt to tend"""
    
    from scipy.integrate import ode
    from numpy import zeros, array, complex
    from expect import expect_comp
    
    #L=L.todense()
    
    t0 = 0
    r = ode(_intfunc).set_integrator('zvode', method='bdf', atol=atol, rtol=rtol)
    r.set_initial_value(initial, t0).set_f_params(L)
    output = Results()
    # Record initial values
    output.t.append(r.t)
    output.rho.append(initial)
    ntimes = int(tend/dt)+1
    if progress:
        bar = Progress(ntimes, name='Time evolution under L...', start=1)
    
    if expect_oper == None:
        while r.successful() and r.t < tend:
            output.rho.append(r.integrate(r.t+dt))
            output.t.append(r.t)
            if progress:
                bar.update()
        return output
    else:
        output.expect = zeros((len(expect_oper), ntimes), dtype=complex)
        output.expect[:,0] = array(expect_comp([initial], expect_oper)).flatten()
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+dt)
            output.expect[:,n_t] = array(expect_comp([rho], expect_oper)).flatten()
            output.t.append(r.t)
            output.rho.append(rho)
            n_t += 1
            if progress:
                bar.update()
        return output

def _intfunc(t, y, L):
    return (L.dot(y))

def steady(L, init=None, maxit=1e6, tol=None):
    
    """calculate steady state of L using sparse eignevalue solver"""

    rho = find_gap(L, init, tol=tol, return_ss=True)   

    return rho
    
def find_gap(L, init=None, maxit=1e6, tol=None, return_ss=False, k=10):
    """Calculate smallest set of k eigenvalues of L"""
    
    from numpy import sort
    from scipy.sparse.linalg import eigs
    from operators import tensor, qeye
    from basis import ldim_s, ldim_p
    from expect import expect_comp
    import gc
    
    if tol is None:
        tol = 1e-8
    
    gc.collect()
    if init is None:
        val, rho = eigs(L, k=k, which = 'SM', maxiter=maxit, tol=tol)
    else:
        val, rho = eigs(L, k=k, which = 'SM', maxiter=maxit, v0=init, tol=tol)
    gc.collect()

    #shift any spurious positive eignevalues out of the way
    for count in range(k):
        if val[count]>1e-10:
            val[count]=-5.0

    sort_perm = val.argsort()
    val = val[sort_perm]

    rho= rho[:, sort_perm]
    
    #calculate steady state and normalise
    if (return_ss):
        rho = rho[:,k-1]
        rho = rho/expect_comp([rho], [tensor(qeye(ldim_p), qeye(ldim_s))])
        rho = rho[0,:]
        return rho
    
    else:
        return val

#calculate spectrum of <op1(t)op2(0)> using initial state op2*rho
#op2 should be a matrix in the full permutation symmetric space
#op1 should be an operator 
def spectrum(L, rho, op1, op2, tlist, ncores):
    
    import scipy.fftpack
    import numpy as np
    
    N = len(tlist)
    dt = tlist[1] - tlist[0]
    
    corr = corr_func(L, rho, op1, op2, tlist[-1], dt, ncores)
    

    F = scipy.fftpack.fft(corr)

    # calculate the frequencies for the components in F
    f = scipy.fftpack.fftfreq(N, dt)

    # select only indices for elements that corresponds
    # to positive frequencies
    indices = np.where(f > 0.0)

    omlist  = 2 * np.pi * f[indices]
    spec =  2 * dt * np.real(F[indices])
    
    

    return spec, omlist


def corr_func(L, rho, op1, op2, tend, dt, ncores, op2_big=False):
    
    from basis import setup_op
    if not op2_big:
        op2 = setup_op(op2, ncores)
    init =  op2.dot(rho) # need to define this in expec
    corr = time_evolve(L, init, tend, dt, [op1])
    return corr.expect[0]
