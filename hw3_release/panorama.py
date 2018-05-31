import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    I2x = dx**2
    I2y = dy**2
    Ixy = dx * dy
    Mxx = convolve(I2x, window)
    Mxy = convolve(Ixy, window)
    Myy = convolve(I2y, window)
    M = np.stack([[Mxx,Mxy],[Mxy,Myy]]).transpose(2,3,0,1)
    response = np.linalg.det(M) - k * (np.trace(M,axis1=2,axis2=3)**2)
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    ### YOUR CODE HERE
    feature = patch.ravel()
    mu = np.sum(feature)
    sigma = np.var(feature) ** 0.5
    if sigma == 0:
        sigma = 1
    feature = (feature - mu) / sigma
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    closest = np.min(dists, axis=0)
    closest_idx1 = np.argmin(dists,axis=0)
    dists[np.argmin(dists,axis=0)]=1000
    next_closest = np.min(dists, axis=0)
    matches_idx2 = np.nonzero(closest/(next_closest + 1e-5) <  threshold)
    matches_idx1 = closest_idx1[matches_idx2]
    matches = np.stack([matches_idx1, matches_idx2[0]], axis=1)
    ### END YOUR CODE
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1) # Transform to homogenious coordinates.
    p2 = pad(p2)

    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2,p1)[0]

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])
 
    max_inliers = np.zeros(N)
    max_inliers_num = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    for iter in range(n_iters):
        matched_smaples = np.random.choice(range(N), n_samples, replace=False)
        p1 = matched1[matched_smaples]
        p2 = matched2[matched_smaples]
        H = np.linalg.lstsq(p2,p1)[0]
        inliers = []
        inliers_num = 0
        for j in range(N):
            d = np.linalg.norm(np.dot(matched2[j],H) - matched1[j])
            if d < threshold:
                inliers.append(j)
                inliers_num += 1
        if inliers_num > max_inliers_num:
            max_inliers_num =  inliers_num
            max_inliers = inliers
    print(max_inliers_num)
    p1 = matched1[max_inliers]
    p2 = matched2[max_inliers]
    H = np.linalg.lstsq(p2,p1)[0]
        
    ### END YOUR CODE
    return H, matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector

    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)

    Returns:
        block: 1D array of shape ((h*w*n_bins)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
   
    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    cells = np.zeros((rows, cols, n_bins))
    

    # Compute histogram per cell
    ### YOUR CODE HERE
    for i in range(rows):
        for j in range(cols):
            cells[i, j, :] = np.histogram(theta_cells[i,j], 9, range=(0,180))
    magn = np.linalg.norm(cells)
    block = (cells / magn).flatten()
    ### YOUR CODE HERE
    
    return block
