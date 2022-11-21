convert_rho = None
convert_rho_dic = {}
import numpy as np
from itertools import permutations


def expect_comp(rho_list, ops):
    """Calculate expectation values of operators in ops
    from compressed density matrices in rho_list"""
    
    from operators import expect, vector_to_operator
    from basis import ldim_p, ldim_s
    
    global convert_rho_dic

    # converted densities matrices (different operators may have different number of
    # spins; we need a list of reduced density matrices for each number of spins)
    rhos_converted_dic = {}
    
    output = []
    for op in ops:
        # number of spins in the target rdm
        nrs = int(np.log(op.shape[0]//ldim_p)/np.log(ldim_s))

        if nrs not in convert_rho_dic:
            raise TypeError('need to run setup_convert_rho_nrs({})'.format(nrs))

        # Only convert compressed matrices in rho
        if nrs not in rhos_converted_dic:
            rhos_converted_dic[nrs] = []
            for count in range(len(rho_list)):
                rho_nrs = convert_rho_dic[nrs].dot(rho_list[count])
                rho_nrs = vector_to_operator(rho_nrs)
                rhos_converted_dic[nrs].append(rho_nrs)
        
        one_output = [] 
        rhos_converted = rhos_converted_dic[nrs]
        for count in range(len(rho_list)):
            one_output.append(expect(rhos_converted[count], op))
        output.append(one_output)

    return output

def get_rdm(rho_list, nrs=1, photon=True):
    from operators import expect, vector_to_operator
    from basis import ldim_p, ldim_s
    
    global convert_rho_dic
    rdms = []
    
    if nrs not in convert_rho_dic:
        raise TypeError('need to run setup_convert_rho_nrs({})'.format(nrs))
    for count in range(len(rho_list)):
        rho_nrs = convert_rho_dic[nrs].dot(rho_list[count])
        rho_nrs = vector_to_operator(rho_nrs)
        if not photon:
            # trace out photon
            sdim = ldim_s**nrs
            rho_nrs_noph = rho_nrs[0:sdim,0:sdim]
            for p in range(1, ldim_p):
                pp1 = p + 1
                rho_nrs_noph += rho_nrs[p*sdim:pp1*sdim,p*sdim:pp1*sdim]
            rho_nrs = rho_nrs_noph
        rdms.append(rho_nrs)
    return rdms


def wigner_comp(rho, xvec, yvec):
    """calculate wigner function of central site from density matrix rho
    at a grid of points defined by xvec and yvec"""
    
    
    global convert_rho
    from operators import expect, vector_to_operator
    from qutip import Qobj, wigner, ptrace
    from basis import ldim_p, ldim_s
    rho_small = convert_rho.dot(rho)
        
    rho_small = vector_to_operator(rho_small)
    
    rho_q = Qobj()
    
    rho_q.data = rho_small
    rho_q.dims = [[ldim_p, ldim_s], [ldim_p, ldim_s]]
    
    rho_q =ptrace(rho_q,0)
    
    w = wigner(rho_q, xvec, yvec)
    
    return w



def photon_dist(rho):
    """return diagonals of photon density matrix"""
    
    
    from operators import vector_to_operator
    from qutip import Qobj, ptrace
    from basis import ldim_p, ldim_s
    
    rho_small = convert_rho.dot(rho)
    rho_small = vector_to_operator(rho_small)
    
    rho_q = Qobj()
    
    rho_q.data = rho_small
    rho_q.dims = [[ldim_p, ldim_s], [ldim_p, ldim_s]]
    
    rho_q =ptrace(rho_q,0)
    
    pops = rho_q.data.diagonal()
    
    return pops
   
def setup_convert_rhos_from_ops(ops):
    """Setup conversion matrices required to calculate reduced density matrices for
    each number of spins occurring in operators in ops. Convenient to run instead
    of separate calls to setup_convert_rho_nrs().""" 
    from basis import ldim_p, ldim_s
    for op in ops:
        # number of spins in the target rdm
        nrs = int(np.log(op.shape[0]//ldim_p)/np.log(ldim_s))
        # check hasn't already been calculated
        if nrs in convert_rho_dic:
            continue
        setup_convert_rho_nrs(nrs)

def setup_convert_rho():
    # retained for compatibility 
    global convert_rho
    convert_rho = setup_convert_rho_nrs(1)


def setup_convert_rho_nrs(nrs=1):
    """Calculate matrix for converting from compressed full to nrs site density matrices.
    Result is stored in global dictionary convert_rho_dic with key nrs.
    Must be run before any calculation requiring the reduced density matrix with nrs spin.
    """  
    
    from basis import nspins, ldim_s, ldim_p
    from indices import indices_elements
    from scipy.sparse import lil_matrix

    assert type(nrs) == int, "Argument 'nrs' must be int"
    assert nrs >= 0, "Argument 'nrs' must be non-negative"
    assert nspins >= nrs, "Number of spins in reduced density matrix ({}) cannot "\
            "exceed total number of spins ({})".format(nrs, nspins)

    global convert_rho_dic

    convert_rho_nrs = lil_matrix(((ldim_p*ldim_s**nrs)**2, ldim_p*ldim_p*len(indices_elements)), dtype=np.float64)

    convert_rho_dic[nrs] = convert_rho_nrs

    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(len(indices_elements)):
                element = np.concatenate(([count_p1], indices_elements[count][0:nspins], 
                                          [count_p2], indices_elements[count][nspins:2*nspins]))
                element_index = ldim_p*len(indices_elements)*count_p1 + len(indices_elements)*count_p2 + count
                bra = element[1:nspins+1] # elements for spin bra
                ket = element[nspins+2:2*nspins+2] # elements for spin ket
                diff_arg = np.asarray(bra != ket).nonzero()[0] # indices where bra and ket differ (axis=0)
                diff_num = len(diff_arg) # number of different spin elements
                if diff_num > nrs:
                    continue
                # get elements that differ
                diff_bra = bra[diff_arg]
                diff_ket = ket[diff_arg]
                same = np.delete(bra, diff_arg) # common elements
                # fill all matrix elements in column element_index according to different and same spins
                add_all(nrs, count_p1, count_p2, diff_bra, diff_ket, same, element_index)
    
    convert_rho_nrs = convert_rho_nrs.tocsr()

    return convert_rho_nrs


def add_all(nrs, count_p1, count_p2, bra, ket, same, element_index, s_start=0):
    """Populate all entries in conversion_matrix with row indices associated with permutations of spin values
    |bra> and <ket| and column index element_index according to the number of permutations of spin values in 
    'same'.

    nrs is the number of spins in the target reduced density matrix ('number reduced spins').
    """
    from basis import ldim_s
    if len(bra) == nrs:
        # add contributions from same to rdm at |bra><ket|
        add_to_convert_rho_dic(nrs, count_p1, count_p2,
                               bra, ket, same, element_index)
        return
    # current |bra> too short for rdm, so move element from same to |bra> (and <ket|)
    # iterate through all possible values of spin...
    for s in range(s_start, ldim_s):
        s_index = next((i for i,sa in enumerate(same) if sa==s), None)
        # ...but only act on the spins that are actually in same
        if s_index is None:
            continue
        # extract spin value from same, append to bra and ket
        tmp_same = np.delete(same, s_index)
        tmp_bra = np.append(bra, s)
        tmp_ket = np.append(ket, s)
        # repeat until |bra> and <ket| are correct length for rdm
        add_all(nrs, count_p1, count_p2, tmp_bra, tmp_ket, tmp_same, element_index, s_start=s)

def add_to_convert_rho_dic(nrs, count_p1, count_p2, diff_bra, diff_ket, same, element_index):
    """Populate entry in conversion matrix with row index corresponding to |count_p1><count_p2|
    for the photon state and |diff_bra><diff_ket| for the spin states and column index 
    element_index according to the number of permutations of spin values in 'same.'

    nrs is the number of spins in the target reduced density matrix ('number reduced spins').
    """
    global convert_rho_dic
    # conversion matrix at this nrs
    convert_rho_nrs = convert_rho_dic[nrs]
    # number of permutations of spins in same, each of which contributes one unit 
    combinations = _multinominal(np.bincount(same))
    # get list of row indices combinations should apply for (all permutations of spins)
    row_indices = get_all_row_indices(count_p1, count_p2, diff_bra, diff_ket)
    for row_index in row_indices:
        convert_rho_nrs[row_index, element_index] = combinations

def get_all_row_indices(count_p1, count_p2, spin_bra, spin_ket):
    """Get all row indices of the conversion matrix corresponding to |count_p1><count_p2|
    for the photon state and |diff_bra><diff_ket| for the spin states."""
    assert len(spin_bra)==len(spin_ket)
    nrs = len(spin_bra)
    s_indices = np.arange(nrs)
    row_indices = []
    for perm_indices in permutations(s_indices):
        index_list = list(perm_indices)
        row_indices.append(get_rdm_index(count_p1, count_p2,
                                         spin_bra[index_list],
                                         spin_ket[index_list]))
    return row_indices


def get_rdm_index(count_p1, count_p2, spin_bra, spin_ket):
    """Calculate row index in conversion matrix for element |count_p1><count_p2| for the 
    photon part and |spin_bra><spin_ket| for the spin part.

    This index is according to column-stacking convention used by qutip - see for example

    A=qutip.Qobj(numpy.arange(4).reshape((2, 2))
    print(qutip.operator_to_vector(A))
    """
    from basis import ldim_p, ldim_s
    bra = np.concatenate(([count_p1],spin_bra))
    ket = np.concatenate(([count_p2],spin_ket))
    row = 0
    column = 0
    nrs = len(bra)-1
    for i in range(nrs+1):
        j = nrs-i
        row += bra[j] * ldim_s**i
        column += ket[j] * ldim_s**i
    return row + column * ldim_p * ldim_s**nrs

def _multinominal(bins):
    """calculate multinominal coeffcient"""
    
    from math import factorial
    
    n = sum(bins)
    combinations  = factorial(n)
    for count_bin in range(len(bins)):
        combinations = combinations//factorial(bins[count_bin])
    return combinations
    
        
        
        

