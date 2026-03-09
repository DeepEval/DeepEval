import GPy
import numpy as np

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    kernels = []
    
    for i in range(n_fidelities):
        if i == 0:
            kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
        else:
            prev_kernel = kernels[i - 1]
            current_kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
            kernel = prev_kernel * current_kernel + base_kernel_class(input_dim=n_input_dims, ARD=ARD)
        
        kernels.append(kernel)
    
    return kernels

if __name__ == "__main__":
    n_fidelities = 3
    n_input_dims = 2
    kernels = make_non_linear_kernels(GPy.kern.RBF, n_fidelities, n_input_dims, ARD=True)
    
    for i, kernel in enumerate(kernels):
        print(f"Kernel for fidelity level {i + 1}:")
        print(kernel)