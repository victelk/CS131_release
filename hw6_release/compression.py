import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    
    H, W = image.shape
    U, s, Vh = np.linalg.svd(image)
#     S = np.zeros_like(image)
#     k = np.arange(num_values)
#     S[k,k] = s[:num_values]
#     compressed_image = np.matmul(np.matmul(U, S), Vh)
    compressed_image = np.dot(U[:,:num_values], np.dot(np.diag(s[:num_values]), Vh[:num_values, :]))
    
    compressed_size = num_values * (image.shape[0] + image.shape[1] + 1)
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
