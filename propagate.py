class Results:
    def __init__(self):
        self.rho= []
        self.t = []
        self.expect = []


def time_evolve(L, initial, tend, dt, expect_oper=None):
    """time evolve matrix L from initial condition initial with step dt to tend"""
    
    from scipy.integrate import ode
    from numpy import zeros, array, complex
    from expect import expect_comp
    
    #L=L.todense()
    
    t0 = 0
    r = ode(_intfunc).set_integrator('zvode', method='bdf', atol=1e-5, rtol=1e-5)
    r.set_initial_value(initial, t0).set_f_params(L)
    output = Results()
    


    if expect_oper == None:
        while r.successful() and r.t < tend:
            output.rho.append(r.integrate(r.t+dt))
            output.t.append(r.t)
        return output
    else:
        ntimes = int(tend/dt)
        n_t=0
        output.expect = zeros((len(expect_oper), ntimes), dtype=complex)
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+dt)
            output.expect[:,n_t] = array(expect_comp([rho], expect_oper)).flatten()
            output.t.append(r.t)
            n_t += 1
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
