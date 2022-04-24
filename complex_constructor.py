import numpy as np
import gudhi
import itertools as it
from functools import partial
from collections import defaultdict

import linear_algebra as lin_alg
import simplex_constructor as sim_constr


def face_types(simplex, compl):
    '''Returns a dictionary that counts how many codim-1 faces of different filtration values simplex has.
   
    simplex = a list of integers such that when leaving out any element, one has a simplex of compl
    compl = gudhi.SimplexTree
    '''
    number_vertices = len(simplex)
    filtration_count = defaultdict(int) # for unassigned keys, this outputs 0 
    for face in it.combinations(simplex, number_vertices-1):
        # iterate through all codim-1 faces
        filtration_count[compl.filtration(list(face))] += 1 # counts 1 for the filtration value of face
    return filtration_count

def has_faces_of_standard(face_dictionary, dimension):
    ''' Checks whether face_dictionary has the correct entries for a simplex of type standard in dimension.
    
    face_dictionary = a dictionary, either via defaultdict or needs integer entries for all possible filtration values
    dimension = an integer
    '''
    if face_dictionary[0] != dimension + 1:
        # all faces need to be of standard type
        return False
    return True  

def has_faces_of_2additive(face_dictionary, dimension):
    ''' Checks whether face_dictionary has the correct entries for a 2-additive simplex in dimension.
    
    face_dictionary = a dictionary, either via defaultdict or needs integer entries for all possible filtration values
    dimension = an integer
    '''
    if face_dictionary[0] != 3:
        # needs 3 face of standard type
        return False
    if face_dictionary[1] != dimension - 2:
        # remaining faces are 2-additive
        return False
    return True  

def has_faces_of_3additive(face_dictionary, dimension):
    ''' Checks whether face_dictionary has the correct entries for a 3-additive simplex in dimension.
    
    face_dictionary = a dictionary, either via defaultdict or needs integer entries for all possible filtration values
    dimension = an integer
    '''
    if face_dictionary[0] != 4:
        # needs 4 face of standard type
        return False
    if face_dictionary[2] != dimension - 3:
        # remaining faces are 3-additive
        return False
    return True

def has_faces_of_double_triple(face_dictionary, dimension):
    ''' Checks whether face_dictionary has the correct entries for a simplex of type double-triple in dimension.
    
    face_dictionary = a dictionary, either via defaultdict or needs integer entries for all possible filtration values
    dimension = an integer
    '''
    if face_dictionary[1] != 4:
        # needs 4 faces of type 2-additive
        return False
    if face_dictionary[2] != 1:
        # needs 1 face of type 3-additive
        return False
    if face_dictionary[3] != dimension - 4:
        # remaining faces are of type double-triple
        return False
    return True

def has_faces_of_double_double(face_dictionary, dimension):
    ''' Checks whether face_dictionary has the correct entries for a simplex of type double-double in dimension.
    
    face_dictionary = a dictionary, either via defaultdict or needs integer entries for all possible filtration values
    dimension = an integer
    '''
    if face_dictionary[1] != 6:
        # needs 6 faces of type 2-additive
        return False
    if face_dictionary[4] != dimension - 5:
        # remaining faces are of type double-double
        return False
    return True


def BAA(A, max_dimension = 7):
    '''Builds BAA, returns it as a gudhi SimplexTree.
    
    A = list of elements in Z^n, given as numpy arrays
    max_dimension = an integer
    '''
    I = gudhi.SimplexTree()
    v = np.arange(len(A)) #vertex labels
    i = 0
    
    # First add the vertices of the complex:
    # print('\n Now building dimension '+str(i)+'-simplices.')
    iSimplex = it.combinations(v, i+1)
    for y in iSimplex:
        x = [A[index] for index in y]
        if sim_constr.spans_standard_simplex(x):
            I.insert(y, 0)
    i += 1
    
    vertex_list = []
    for vertex in I.get_skeleton(0):
        vertex_list.append(vertex[0][0])
    
    # Now add higher dimensional simplices
    while i < min(len(A),max_dimension+1):
        # print('\n Now building dimension '+str(i)+'-simplices.')        
        
        # First generate a list of all potential simplices (these are typically way less then the (i+1)-element 
        # subsets of vertex_list).
        potential_i_simplices = []
        for simplex in I.get_skeleton(i-1):
            # important here: .get_skeleton() outputs simplices in lexicographic ordering
            codim_1_simplex = simplex[0]
            if len(codim_1_simplex) == i:
                # only then it has codimension 1
                for vertex in vertex_list:
                    if vertex > codim_1_simplex[i-1]:
                        # if this is not true, then the case was already handled before (use lex. ordering)
                        potential_simplex = list(codim_1_simplex)
                        potential_simplex.append(vertex)
                        # now check whether all faces of potential_simplex are in the complex
                        faces_in = True
                        for face in it.combinations(potential_simplex, i):
                            if not I.find(face):
                                faces_in = False
                                break
                        if faces_in:
                            potential_i_simplices.append(potential_simplex)
        # print('Number of potential '+str(i)+'-simplices: '+str(len(potential_i_simplices)))                    
            
        for y in potential_i_simplices:
            face_dictionary = face_types(y, I)
            
            x = [A[index] for index in y]
            if has_faces_of_standard(face_dictionary, i):
                # first check whether y has the correct faces for being such a simplex
                if sim_constr.spans_standard_simplex(x):
                    I.insert(y, 0)
                    #birth time 0 for standard smplices
                    continue
            if has_faces_of_2additive(face_dictionary, i):
                # first check whether y has the correct faces for being such a simplex
                if i != 2:
                    # for everthing above dimension 2, it's sufficient to know that the faces are correct
                    I.insert(y, 1)
                    #birth time 1 for 2-additive simplices
                    continue
                if sim_constr.spans_2additive_simplex(x):
                    I.insert(y, 1)
                    #birth time 1 for 2-additive simplices
                    continue
            if has_faces_of_3additive(face_dictionary, i):
                # first check whether y has the correct faces for being such a simplex
                if i != 3:
                    # for everthing above dimension 3, it's sufficient to know that the faces are correct
                    #birth time 2 for the 3-additive simplices
                    I.insert(y,2)
                    continue
                if sim_constr.spans_3additive_simplex(x):
                    #birth time 2 for the 3-additive simplices
                    I.insert(y,2)
                    continue
            if has_faces_of_double_triple(face_dictionary, i):
                # for these simplices, it's sufficient to know that the faces are correct
                #birth time 3 for the double-triple simplices
                I.insert(y,3)
                continue
            if has_faces_of_double_double(face_dictionary, i):
                # for these simplices, it's sufficient to know that the faces are correct
                #birth time 4 for the double-double simplices
                I.insert(y,4)
                continue
        i += 1
    # print('\n Finished building the complex.')
    return I
    
    
def link_BAA(vertices_link, vertices_sigma):
    '''Computes the link of sigma in the subcomplex of BAA spanned by vertices_link and vertices_sigma.
    
    vertices_link = a list of elements in Z^n (as np.array)
    vertices_sigma = a list of elements in Z^n (as np.array)
        should form a simplex in BAA
    The lists should be disjoint(with elements seen as vertices of B).
    '''
    size_link = len(vertices_link)
    size_sigma = len(vertices_sigma)
    vertices_star = []
    vertices_star.extend(vertices_link)
    vertices_star.extend(vertices_sigma)
    subcomplex = BAA(vertices_star) # this is the subcomplex of BAA spanned by vertices_star
    # When constructing the complexes inductively over the skeleta, this needs to be computed
    # if one wants to build the link.
    # Now compute the link of sigma in this complex. (sigma corresponds to vertex number_vertices-1 in the simplex tree)
    sigma = list(range(size_link,size_link+size_sigma))
    link = gudhi.SimplexTree()
    for coface in subcomplex.get_star(sigma):
        face = coface[0]
        for i in range(size_sigma):
            # remove the last items of the face, thes are the vertices of sigma
            face.pop() 
        link.insert(face, coface[1])
        # Attention: The filtration value added here is the one of face\ast sigma. 
        # This is not necessarily the same as the one of face. (face might be standard while face\ast sigma is additive e.g.) 
        # This coincides with the types of simplices in Linkhat that is defined in the article.
    return link
