import numpy as np
import scipy.sparse as sp

#heavily influenced by qutip

def qeye(N):

    return sp.eye(N, N, dtype=complex, format='csr')

def destroy(N):

    return sp.spdiags(np.sqrt(range(0, N)),
                           1, N, N, format='csr')

def create(N):

    qo = destroy(N)  # create operator using destroy function
    qo = qo.T.tocsr()  # transpose data in Qobj and convert to csr
    return qo

def sigmap():
    return sp.spdiags(np.array([0.0,1.0]),
                      1, 2, 2, format='csr')
                      
def sigmam():
    return sigmap().T.tocsr()

def sigmaz():
    return sp.spdiags(np.array([1.0,-1.0]),
                      0, 2, 2, format='csr')
def sigmax():
    return sigmap() + sigmam()

def sigmay():
    return -1j*sigmap() + 1j*sigmam()
    
def tensor(*args):

    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    elif len(args) == 1 and isinstance(args[0], Qobj):
        # tensor is called with a single input, do nothing
        return args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args


    out = []

    for n, q in enumerate(qlist):
        if n == 0:
            out = q
        else:
            out = sp.kron(out, q, format='csr')

    return out
    
def basis(N, n=0):
    
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    if (not isinstance(n, (int, np.integer))):
        raise ValueError("n must be integer n >= 0")

    if n > (N - 1):  # check if n is within bounds
        raise ValueError("basis vector index need to be in n <= N-1")

    bas = sp.lil_matrix((N, 1))  # column vector of zeros
    bas[n, 0] = 1  # 1 located at position n
    bas = bas.tocsr()

    return tensor(bas, bas.T)
    
def expect(oper, state):

    # calculates expectation value via TR(op*rho)
    return (oper.dot(state).toarray()).trace()

def vector_to_operator(op):

    n = int(np.sqrt(op.shape[0]))
    q = sp_reshape(op.T, (n, n)).T
    return q
    
def sp_reshape(A, shape, format='csr'):

    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('Shape must be a list of two integers')

    C = sp.coo_matrix(A)
    nrows, ncols = C.shape
    size = nrows * ncols
    new_size = shape[0] * shape[1]

    if new_size != size:
        raise ValueError('Total size of new array must be unchanged.')

    flat_indices = ncols * C.row + C.col
    new_row, new_col = divmod(flat_indices, shape[1])
    B = sp.coo_matrix((C.data, (new_row, new_col)), shape=shape)

    if format == 'csr':
        return B.tocsr()
    elif format == 'coo':
        return B
    elif format == 'csc':
        return B.tocsc()
    elif format == 'lil':
        return B.tolil()
    else:
        raise ValueError('Return format not valid.')


