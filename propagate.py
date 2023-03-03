class Results:
    def __init__(self):
        self.rho= []
        self.t = []
        self.expect = []

class Progress:
    def __init__(self, total, description='', start_step=0):
        self.description = description
        self.step = start_step
        self.end = total-1
        self.percent = self.calc_percent()
        self.started = False

    def calc_percent(self):
        return int(100*self.step/self.end)

    def update(self, step=None):
        # print a description at the start of the calculation
        if not self.started:
            print('{}{:4d}%'.format(self.description, self.percent), end='', flush=True)
            self.started = True
            return
        # progress one step or to the specified step
        if step is None:
            self.step += 1
        else:
            self.step = step
        percent = self.calc_percent()
        # only waste time printing if % has actually increased one integer
        if percent > self.percent:
            print('\b\b\b\b\b{:4d}%'.format(percent), end='', flush=True)
            self.percent = percent
        if self.step == self.end:
            print('', flush=True)

def time_evolve(L, initial, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5, progress=False):
    """time evolve matrix L from initial condition initial with step dt to tend"""
    
    from scipy.integrate import ode
    from numpy import zeros, array
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
        bar = Progress(ntimes, description='Time evolution under L...', start_step=1)
    
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

    rho = find_gap(L, init, maxit, tol, return_ss=True)   

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
    # pfw: use ARPACK shift-invert mode to find eigenvalues near 0
    val, rho = eigs(L, k=k, sigma=0, which = 'LM', maxiter=maxit, v0=init, tol=tol)
    # N.B. unreliable to find mutliple eigenvalues, see https://github.com/scipy/scipy/issues/13571
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
