indices_elements = []
indices_elements_inv = {}

# Sets up two dictionaries:
""" 1) indices_elements maps from a reduced density matrix index to a vector 
with the full set indices for that element
 e.g for 3 spins this could be [s00 s10 s20 s01 s11 s21]
 2) indices_elements_inv maps from a TUPLE containing a list of 
 dm indices to an elemnet in the compressed space

THESE DICTIONARIES NOW DO NOT INCLUDE THE PHOTON STATE
The photon state can be caluated using the fact that the full dm is justy the tensor 
product of the photon with the compressed list of spin indices"""

def list_equivalent_elements():
    """Generate basis list, needs to be run at the beginning of
    each calculation"""
    
    
    global indices_elements, indices_elements_inv
    
    from basis import nspins, ldim_p, ldim_s
    from numpy import concatenate, copy, array

    indices_elements = []
    indices_elements_inv = {}
    
    count = 0
    
    #get minimal list of left and right spin indices (in combined form)
    spins = setup_spin_indices(nspins)
    
    left =[]
    right = []
    
    #split combined indices into left/right form
    for count in range(len(spins)):
        leftadd, rightadd = _to_hilbert(spins[count])
        left.append(leftadd)
        right.append(rightadd)

    
    left = array(left)
    right = array(right)

    #loop over each photon state and each spin configuration
    for count in range(len(spins)):
                
                #calculate element and index 
                element = concatenate((left[count], right[count]))
                
                #add appropriate entries to dictionaries
                indices_elements.append(copy(element))
                indices_elements_inv[_comp_tuple(element)] = count


def setup_spin_indices(ns):
    """get minimal list of left and right spin indices"""
    
    from basis import ldim_s
    from numpy import concatenate, array, copy
    
    spin_indices = []
    spin_indices_temp = []
    
    #construct all combinations for one spin
    for count in range(ldim_s**2):
        spin_indices_temp.append([count])
    spin_indices_temp = array(spin_indices_temp)
    
    #loop over all other spins
    for count in range(ns-1):
        #make sure spin indices is empty 
        spin_indices = []   
        #loop over all states with count-1 spins
        for index_count in range(len(spin_indices_temp)):
         
            #add all numbers equal to or less than the last value in the current list
            for to_add in range(spin_indices_temp[index_count, -1]+1):
                spin_indices.append(concatenate((spin_indices_temp[index_count, :], [to_add])))
        spin_indices_temp = copy(spin_indices)
    
    return spin_indices
        
    
    

def _index_to_element(index, ns= None):
    """convert a combined spin index to an element with ns spins
    NOT for convering from a full dm index to an element"""
    
    from basis import nspins, ldim_s
    
    if ns == None:
        ns = nspins
    element = []
    
    #do appropriate modulo arithmatic 
    for count in range(ns):
        element.append(index%ldim_s)
        index = (index - element[-1])//ldim_s
    return element
        
            

def get_equivalent_dm_tuple(dm_element):
    """calculate tuple representation of dm element which is equivalent to dm_element"""
    
    from basis import nspins,ldim_s, ldim_p
    from numpy import sort, concatenate,array
    
    if len(dm_element) != 2*(nspins):
        raise TypeError('dm_index has the wrong number of elements')
    
    left = array(dm_element[0:nspins])
    right = array(dm_element[nspins:2*nspins])
    
    #use combined Hilbert space indices for both left and right vectors
    combined = _full_to_combined(left, right)
    #The equivalent element is one in which this list is sorted
    combined = -sort(-1*combined)


    newleft, newright = _combined_to_full(combined)
    dm_element_new = concatenate((newleft, newright))
  
    #return the index of the equvalent element
    return _comp_tuple(dm_element_new)


def _to_hilbert(combined):
    """convert to Hilbert space index"""
    
    left = []
    right = []
    for count in range(len(combined)):
        leftadd, rightadd = _combined_to_full(combined[count])
        left.append(leftadd)
        right.append(rightadd)
    return left, right
    

def _combined_to_full(combined):
    """create left and right Hilbert space indices from combined index"""
    
    from basis import ldim_s    
    right = combined%ldim_s
    left = (combined - right)//ldim_s
    
    return left, right


def _full_to_combined(left, right):
    """create left and right Hilbert space indices from combined index"""
    from basis import ldim_s
    return ldim_s*left + right


def _comp_tuple(element):
    """compress the tuple used in the dictionary"""
    from basis import nspins, ldim_s, ldim_p
    
    element_comp = []
    
    for count in range(nspins):
        element_comp.append(element[count]*ldim_s + element[count+nspins])
    return tuple(element_comp)




    

    
