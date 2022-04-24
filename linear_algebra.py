import numpy as np
import scipy.linalg as spla
import itertools as it

from math import gcd
from functools import reduce



def gcd_array(array):
    ''' Returns the gcd of array.
    source: https://stackoverflow.com/questions/29194588/python-gcd-for-list
    
    array = a one dimensional array or list of integers
    '''
    
    x = reduce(gcd, array)
    #the following makes sure that gcd is applied at least once
    x = gcd(x,0)
    return x
    

# --- is_partial_basis ---
# function: determines if a set of m vectors spans a rank m summand of Z^n, i.e. whether it is a partial basis
# input: matrix (n,m)
# output: boolean
# Source: https://math.stackexchange.com/questions/2154381/how-to-check-if-elements-form-a-primitive-system-of-a-lattice


def gcd_minor_dets_invertible(A):
    ''' Determines if the gcd of the determinants of all max-dimensional minors of A is invertible in Z.
    
    A = a matrix (numpy)
    '''
    k = len(A) # number of vectors
    
    if k > len(A[0]):
        raise Exception("Minors do not exist.") 
        
    minors = it.combinations(np.transpose(A), k)
    current_gcd = 0
    for minor in minors:
        current_gcd = gcd(current_gcd, int(round(np.linalg.det(minor))))
        if current_gcd == 1:
            return True
    return False

def is_partial_basis(A):

    if len(A) > len(A[0]):
        return False
        
    if np.linalg.matrix_rank(A) != len(A): # the set of vectors needs to be linearly independent
        return False
    #redundant because of next, but speeds up
    
    return gcd_minor_dets_invertible(A)
    


def is_additive_k_tuple(A):
    '''Given k > 2 vectors, does there exist an additive linear 
    dependency of the form v_k = +/- v_1 +/- ... +/- v_{k-1}? Returns True if yes, False otherwise.
    
    A = np.array, rows are the vectors that are checked for linear dependency.
    '''

    k = len(A)
    
    if k < 3:
        return False # A is too small to contain additive relations
   
    first_rows = A[:k-1]
    last_row = A[k-1]
    
    coefficients, residual, rank, SVs = np.linalg.lstsq(np.transpose(first_rows),last_row,rcond=None)
    
    if rank != k-1:
        return False
        
    except_as_zero = np.finfo(float).eps*max(SVs)*k*100
    
    if len(residual) != 0:
        if not residual[0] < except_as_zero:
            # then last_row cannot be written as a sum of first_rows
            return False
    else:
        # then we need to calculate the residual by hand
        if not np.linalg.norm(np.transpose(first_rows) @ coefficients - last_row) < except_as_zero:
            # then last_row cannot be written as a sum of first_rows
            return False
    
    for coefficient in coefficients:
        if round(abs(coefficient)) !=1:
            return False

    return True
