import numpy as np
import itertools as it

import linear_algebra as lin_alg

def spans_standard_simplex(list_of_vectors):
    '''Determines whether list_of_vectors determines a standard simplex. Returns boolean.
    
    list_of_vectors = list of vectors in Z^n, e.g. as lists of integers    
    '''

    A = np.array(list_of_vectors)

    if not lin_alg.is_partial_basis(A): # the set needs to span a summand of Z^n
        return False
    
    return True


def spans_2additive_simplex(list_of_vectors):
    '''Determines whether list_of_vectors determines a 2-additive simplex. Returns boolean.
    
    list_of_vectors = list of vectors in Z^n, e.g. as lists of integers    
    '''

    A = np.array(list_of_vectors)

    if (len(A) > 2 and # there are at least 3 vectors
        np.linalg.matrix_rank(A) == len(A)-1): # linear dependency detected

        for x in it.combinations(range(len(A)), 3):
            T = A[np.array(x)]
            if (lin_alg.is_additive_k_tuple(T)
                and lin_alg.is_partial_basis(np.delete(A, x[2], 0))):
                return True
        
    return False 
      
def spans_3additive_simplex(list_of_vectors):
    '''Determines if a set of m vectors spans a 3-additive simplex. Returns boolean.
    
    list_of_vectors = list of vectors in Z^n, e.g. as lists of integers  
    '''

    A = np.array(list_of_vectors)

    if (len(A) > 3 and # there are at least 4 vectors.
        np.linalg.matrix_rank(A) == len(A)-1): # one linear depency detected (can't span a standard simplex)

        for x in it.combinations(range(len(A)), 4):
            T = A[np.array(x)] # select 4 row vectors
            if (lin_alg.is_additive_k_tuple(T)
                and lin_alg.is_partial_basis(np.delete(A, x[3], 0))): # if the linear dependency is deleted, we get a standard simplex
                return True
        
    return False
