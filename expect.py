convert_rho = None


def expect_comp(rho, ops):
    """Calculate expectation values of operators in ops
    from a compressed density matrix rho"""
    
    from operators import expect, vector_to_operator
    from basis import ldim_p, ldim_s
    
    global convert_rho
    
    if convert_rho is None:
        raise TypeError('need to run setup_convert_rho()')

    output = []
    for count_ops in range(len(ops)):
        output.append([])
    
    for count in range(len(rho)):
        
        #get single site density matrix by multiplying with conversion matrix
        rho_ss = convert_rho.dot(rho[count])
        
        rho_ss = vector_to_operator(rho_ss)

        for count_ops in range(len(ops)):
            output[count_ops].append(expect(rho_ss, ops[count_ops]))
    return output


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
    

def setup_convert_rho():
    
    """calculate matrix for converting from compressed full to single site density matrices
    This function needs to be run at the start of each calculation"""  
    
    from basis import nspins, ldim_s, ldim_p
    from indices import  indices_elements
    from numpy import concatenate, copy, array_equal, sort, bincount, argmax, float64
    from scipy.sparse import lil_matrix
    
    global convert_rho
        
    convert_rho = lil_matrix(((ldim_p*ldim_s)**2, ldim_p*ldim_p*len(indices_elements)), dtype = float64)
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(len(indices_elements)):

                element = concatenate(([count_p1], indices_elements[count][0:nspins], [count_p2], indices_elements[count][nspins:2*nspins]))
                element_index = ldim_p*len(indices_elements)*count_p1 + len(indices_elements)*count_p2 + count
                left = element[0:nspins+1]
                right = element[nspins+1:2*nspins+2]
        
                #check if there is only one off diagonal element
                diff = sort(abs(left[1:] - right[1:]))
        
                if (diff[-2] == 0 and diff[-1]>0):
            
                    #find location of off-diagonal element
                    loc = argmax(abs(left[1:] - right[1:]))
            
                    #shift to spin0
                    left[1], left[loc+1] = left[loc+1], left[1]
                    right[1], right[loc+1] = right[loc+1], right[1]
            
                    small_element = copy(left[2:])
                    #bin element so that [1,0,0,1,1] = [2,3]
                    ldim_counts = bincount(small_element)
                    combinations  = _multinominal(ldim_counts)
            
                    #calculate current position in column stacked dm
                    column_index = left[0]*ldim_s + left[1] + ldim_s*ldim_p*(right[0]*ldim_s + right[1])
                    convert_rho[column_index, element_index] = combinations
        
                #check if element is diagonal in all space except photon
                if array_equal(left[1:], right[1:]):
            
                    #calculate number of combinations for unchanged element
                    small_element = copy(left[2:])
                    ldim_counts = bincount(small_element)
                    combinations  = _multinominal(ldim_counts)

            
                    #calculate current position in column stacked dm
                    column_index = left[0]*ldim_s + left[1] + ldim_s*ldim_p*(right[0]*ldim_s + right[1])
                    convert_rho[column_index, element_index] = combinations
            

            
                    #make sure that the element in spin zero is accounted for
                    if len(ldim_counts<left[1]+1):
                        ldim_counts.resize([left[1]+1])
                    #swap spin 0 for other spins and work out combinations
                    for ii in range(len(ldim_counts)):
                        #only swap if there is a spin to swap with
                        if ldim_counts[ii] > 0:
                        
                            ldim_counts_temp = copy(ldim_counts)
                            ldim_counts_temp[ii] = ldim_counts_temp[ii] - 1
                            ldim_counts_temp[left[1]] = ldim_counts_temp[left[1]] + 1
                            combinations  = _multinominal(ldim_counts_temp)
                            column_index = left[0]*ldim_s + ii + ldim_s*ldim_p*(right[0]*ldim_s + ii)
                            convert_rho[column_index, element_index] = combinations
            
    convert_rho = convert_rho.tocsr()



def _multinominal(bins):
    """calculate multinominal coeffcient"""
    
    from math import factorial
    
    n = sum(bins)
    combinations  = factorial(n)
    for count_bin in range(len(bins)):
        combinations = combinations//factorial(bins[count_bin])
    return combinations
    
        
        
        

