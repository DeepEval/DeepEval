import torch

def winger_D(k, alpha, beta, gamma):
    """Compute Wigner D matrix representation of the SO(3) group"""
    # Implement your code here
    # ...

    # Return the Wigner D matrix
    return D_mat

if __name__ == "__main__":
    # Sample input values
    k = 2
    alpha = torch.tensor(0.5)
    beta = torch.tensor(0.25)
    gamma = torch.tensor(0.125)

    # Call the function
    D_mat = winger_D(k, alpha, beta, gamma)

    # Print the results
    print(D_mat)