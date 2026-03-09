import torch
import numpy as np

def function_name(k):
    # Define the parameters for the SO(3) generators
    theta = torch.randn(k) * 2 * np.pi  # Random theta values
    phi = torch.randn(k) * np.pi  # Random phi values
    r = torch.randn(k)  # Random r values

    # Convert the SU(2) generators to the SO(3) basis
    def su2_to_so3(theta, phi, r):
        # Convert the angles to Euler angles
        alpha = torch.atan2(r, torch.tan(phi / 2))
        beta = torch.atan2(torch.sqrt((1 - torch.cos(theta)) / (1 + torch.cos(theta))), torch.tan(phi / 2))
        gamma = torch.atan2(theta, torch.tan(phi / 2))

        # Convert the Euler angles to a rotation matrix
        R_beta = torch.tensor([[torch.cos(beta), -torch.sin(beta), 0],
                               [torch.sin(beta), torch.cos(beta), 0],
                               [0, 0, 1]])
        R_gamma = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                                [torch.sin(gamma), torch.cos(gamma), 0],
                                [0, 0, 1]])
        R_alpha = torch.eye(3)
        R = R_alpha.matmul(R_beta).matmul(R_gamma)

        return R

    # Generate the SO(3) generators
    J_x = su2_to_so3(theta, phi, r)
    J_y = su2_to_so3(theta, phi, r)
    J_z = su2_to_so3(theta, phi, r)

    # Stack the generators
    J_stack = torch.stack([J_x, J_y, J_z])

    return J_stack

if __name__ == "__main__":
    # Create sample input values
    k = 3
    input_values = torch.randn(k, 3)

    # Call the function and print the results
    results = function_name(k)
    print("SO(3) Generators:")
    print(results)