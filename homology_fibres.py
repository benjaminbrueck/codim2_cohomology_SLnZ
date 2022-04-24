import numpy as np
import random
import itertools as it
import copy

import linear_algebra as lin_alg
import complex_toolbox as com_tools

def vertices_fibre(sigma,w):
    '''Compute the vertices in the fibre of sigma in BAA^{<R} under the map that send w to 0.

    sigma = a list of elements in Z^n (as np.array)
    w = an element in Z^n (as np.array), with positive last coordinate
    '''
    # in the following, assume that the last coordinate of w is positive
    dim = len(w)
    R = w[dim-1]
    vertices_in_fibre = []
    for v in sigma:
        v_new = v
        while v_new[dim-1] < R:
            if abs(v_new[dim-1]) < R and lin_alg.gcd_array(v_new) == 1:
                vertices_in_fibre.append(v_new)
            v_new = v_new + w
        v_new = v - w
        while v_new[dim-1] > -R:
            if abs(v_new[dim-1]) < R and lin_alg.gcd_array(v_new) == 1:
                vertices_in_fibre.append(v_new)
            v_new = v_new - w
    # Remove possible duplicates in the list
    vertices_in_fibre = com_tools.remove_duplicates(vertices_in_fibre)
    return vertices_in_fibre
    

# Functions for listing isomorphism types of Q complexes

def complete_to_invertible(last_column, max_entry = 5):
    '''Generates an nxn-matrix with integer coefficients between -max_entry and max_entry
    that is invertible over Z and with a specified last_column.
    
    last_column =  a length-n tuple or list of integers
    max_entry = a positive integer
    
    '''
    # super inefficient, but does what it's supposed to do
    dimension = len(last_column)
    M = np.zeros((dimension,dimension), dtype=int)
    for i in range(dimension):
        M[i][dimension-1] = last_column[i]

    for counter in range(100000):
        # fill the matrix at random until one finds something invertible
        for i in range(dimension):
            for j in range(dimension-1):
                M[i][j] = random.randint(-max_entry,max_entry)
        det = int(round(np.linalg.det(M)))
        if det == 1 or det == -1:
            return(M)
    print('Did not find anything')
    return False
    
    
def potential_last_entries(max_value,dimension = 4):
    '''Lists all dimension-tuples of integers (a1,...,a_{dimension-1},R) such that R<=max_value
    and |ai|<R for all i.
    
    max_value = a positive integer
    '''
    iso_types = []
    for last_entries in it.product(range(-max_value,max_value+1), repeat = dimension):
        if last_entries[dimension-1]<=0:
            continue
        valid = True
        for entry in last_entries[:dimension-1]:
            if not abs(entry) < abs(last_entries[dimension-1]):
                valid = False
                break
        if not valid:
            continue
        if lin_alg.gcd_array(last_entries) == 1:
            iso_types.append(last_entries)
    return iso_types

def Rint(number,R):
    '''Computes the R-interval of number.
    Returns n if: 
        n is even and number = n/2R 
        n is odd and number is in the interval ((n-1)/2R, (n+1)/2*R)
    
    number = an integer
    R = a positive integer
    '''
    if (number % R) == 0:
        return int(number/R)*2
    else:
        return int(number/R)*2+np.sign(number)
    
def sign(number):
    '''Outputs the sign of number, where 0 has positive sign
    
    number = an integer    
    '''
    if not number == 0:
        return np.sign(number)
    else:
        return 1

def iso_types(num_ind_par,dep_pars, max_R_int = 10):
    '''Computes a list with all isomorphism types of fibres (complexes Q), encoded via R-intervals.
    
    num_ind_par = positive integer, the number of independent parameters 
    dep_pars = a list with one entry for each dependent parameter
    max_R_int = an integer; ATTENTION: needs to be bigger than the absolute value of any R-interval
                            that can occur
    
    for each dependent parameter, one specifices a list that encode the dependencies of this parameter
    on the previous ones. E.g. entering [[0,1],[2,-3]] means that this dependent parameter is both equal to 
    the sum of the 0th and the 1st parameter and the sum of the 2nd and the negative of the 3rd parameter.
    For this, it doesn't matter if the 0th, 1st or 2nd parameter are independent or not. It's just important
    that all of these parameters were introduced before coming to the parameter [[0,1],[2,-3]] (so here, this
    depend parameter needs to have at least position 4 in the list.)
    
    
    Current restrictions of this function:
    The dependencies can just be of the form v_k = +-v_i +- v_k.
    So just sum of two and just coefficients +-1 possible.
    '''
    #All independet parameters habe R-interval between -1 and 1.
    possible_values_zi = [-1,0,1]
    types_list = []
    for ind_combination in it.product(possible_values_zi, repeat = num_ind_par):
        # Create a list of all possible combinations for the R-intervals of the 
        # independent parameters.
        types_list.append(list(ind_combination))
        
    for parameter in dep_pars:
        # add all possible values of this new parameter to the combinations of previous parameters
        for j in range(len(types_list)):
            # iterate through list that's there so far
            current_type = types_list.pop(0)
            even = False
            for dependency in parameter:
                # check if something has even R-interval
                if (current_type[abs(dependency[0])] % 2) == 0 or (current_type[abs(dependency[1])] % 2) == 0:
                    # if R-interval is even, new_entry is uniquely determined
                    even = True
                    new_type = copy.deepcopy(current_type)
                    new_entry = sign(dependency[0])*current_type[abs(dependency[0])] + sign(dependency[1])*current_type[abs(dependency[1])]
                    new_type.append(new_entry)
                    types_list.append(new_type)
                    break
            if not even:
                # means that there are no even R-intervals here, so there are three possibilites for new_entry
                # now list all values that would be possible by the different dependencies
                poss_values = set(range(-max_R_int,max_R_int+1)) # ad-hoc to make code easier
                for dependency in parameter:
                    poss_dep = set()
                    for i in range(-1,2):
                        poss_dep.add(sign(dependency[0])*current_type[abs(dependency[0])] + sign(dependency[1])*current_type[abs(dependency[1])]+i)
                        # everything that's possible with this dependency
                    poss_values = poss_values.intersection(poss_dep) # compare to possibilities from other dependencies
                for new_entry in poss_values:
                    new_type = copy.deepcopy(current_type)
                    new_type.append(new_entry)
                    types_list.append(new_type)     
    return types_list
