import GPy
import numpy as np

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    kernels = []
    for i in range(n_fidelities):
        kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
        if i > 0:
            prev_kernel = kernels[-1]
            kernel = (kernel * prev_kernel) + base_kernel_class(input_dim=1, ARD=ARD)
        kernels.append(kernel)
    return kernels

if __name__ == "__main__":
    # Create a base kernel class
    base_kernel_class = GPy.kern.RBF
    
    # Define the parameters
    n_fidelities = 3
    n_input_dims = 2
    ARD = False
    
    # Call the function
    kernels = make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD)
    
    # Print the results
    for i, kernel in enumerate(kernels):
        print(f"Kernel {i+1}:")
        print(kernel)
        print()