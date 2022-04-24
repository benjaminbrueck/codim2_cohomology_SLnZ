# functions for generating vertex sets and working with complexes

import gudhi
    
def remove_duplicates(list_of_vectors):
    '''After applying this function, the list contains only one copy of v or -v.
    
    list_of_vectors = a list of numpy arrays    
    '''
    already_there = set()
    list_without_duplicates = []
    for vector in list_of_vectors:
        tupled_vector = tuple(vector)
        if not tupled_vector in already_there:
            list_without_duplicates.append(vector)
            neg_tupled_vector = tuple(-vector)
            already_there.add(tupled_vector)
            already_there.add(neg_tupled_vector)
    return list_without_duplicates


def number_simplices_dimensions(compl):
    ''' Returns a tuple with the number of simplices of compl in different dimensions.
    
    compl = a gudhi SimplexTree
    '''
    
    max_dimension = compl.dimension()
    simplex_count = [0]*(max_dimension + 1)
    for x in compl.get_simplices():
        dimension = len(x[0])-1
        simplex_count[dimension] += 1
    return simplex_count
