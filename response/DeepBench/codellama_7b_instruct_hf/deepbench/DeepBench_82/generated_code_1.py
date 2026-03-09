import numpy as np
from scipy.sparse import coo_matrix

def pixel_to_pixel_connections(n_x, n_y, n_z=1, mask=None, return_as=coo_matrix, dtype=int):
    # Check if mask is provided
    if mask is not None:
        # Make sure the mask has the right shape
        if mask.shape != (n_x, n_y, n_z):
            raise ValueError("The mask must have the same shape as the image")
        # Apply the mask to the image
        image = np.zeros((n_x, n_y, n_z), dtype=bool)
        image[mask] = True
    else:
        image = np.ones((n_x, n_y, n_z), dtype=bool)

    # Create a list of all the pixels in the image
    pixels = [(x, y, z) for x in range(n_x) for y in range(n_y) for z in range(n_z)]

    # Create a list of all the pixel connections
    connections = []
    for i in range(len(pixels)):
        for j in range(i+1, len(pixels)):
            if image[pixels[i][0], pixels[i][1], pixels[i][2]] and image[pixels[j][0], pixels[j][1], pixels[j][2]]:
                connections.append((i, j))

    # Create the adjacency matrix
    adjacency_matrix = coo_matrix((np.ones(len(connections)), (connections)), shape=(len(pixels), len(pixels)))

    # Return the adjacency matrix
    return adjacency_matrix

# Example usage
if __name__ == "__main__":
    # Create a sample image
    image = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    # Create a mask for the image
    mask = np.array([[True, False, True],
                     [True, True, True],
                     [False, True, False]])

    # Compute the pixel-to-pixel connections
    connections = pixel_to_pixel_connections(image.shape[0], image.shape[1], image.shape[2], mask)

    # Print the connections
    print(connections)