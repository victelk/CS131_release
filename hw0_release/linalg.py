import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """
    out = None
    ### YOUR CODE HERE
    out = np.dot(vector1, vector2)
    ### END YOUR CODE

    return out

def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (1, x)
    """
    out = None
    ### YOUR CODE HERE
    out = (vector1.T * vector2) * (M * vector1)
    ### END YOUR CODE

    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    ### YOUR CODE HERE
    u, s, v = np.linalg.svd(matrix)
    ### END YOUR CODE

    return u, s, v

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    u, s, v = svd(matrix)
    ### YOUR CODE HERE
    assert n <= min(matrix.shape), "too many eignenvalues required"
#     singular_values = np.sort(np.diagonal(s))[::-1][:n]
    singular_values = np.diagonal(s)[:n]
    ### END YOUR CODE
    return singular_values

def eigen_decomp(matrix):
    """ Implement Eigen Value Decomposition
    Args:
        matrix: numpy matrix of shape (m,m)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    w = None
    v = None
    ### YOUR CODE HERE
    v, w = np.linalg.eig(matrix) 
    ### END YOUR CODE
    return w, v

def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return
        
    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    print(matrix)
    w, v = eigen_decomp(matrix)
    print(v)
    eigen_values = []
    eigen_vectors = []
    ### YOUR CODE HERE
    print(matrix.shape[0])
    assert num_values <= matrix.shape[0], "too many eignenvalues required"
    ind = np.argsort(v)[::-1][:num_values]
    eigen_values = v[ind]
    eigen_vectors = w[:, ind]
    ### END YOUR CODE
    return eigen_values, eigen_vectors
