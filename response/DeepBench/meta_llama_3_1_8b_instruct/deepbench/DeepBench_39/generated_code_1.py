import torch
import torch.nn as nn
import math

def so3_generators(k):
    # Define the generators of the SU(2) algebra
    su2_generators = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0]
    ])

    # Define the transformation matrix to convert from SU(2) to SO(3)
    su2_to_so3 = torch.tensor([
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, -1]
    ])

    # Calculate the generators of the SO(3) algebra
    so3_generators = torch.mm(su2_to_so3, su2_generators)

    # Split the generators into J_x, J_y, and J_z
    j_x = so3_generators[:, :3]
    j_y = so3_generators[:, 3:6]
    j_z = so3_generators[:, 6:]

    # Stack the generators
    generators = torch.stack((j_x, j_z, j_y), dim=0)

    return generators

if __name__ == "__main__":
    # Create a sample input value
    k = 1

    # Call the function
    generators = so3_generators(k)

    # Print the results
    print(generators)