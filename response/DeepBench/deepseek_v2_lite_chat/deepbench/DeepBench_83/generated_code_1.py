import numpy as np
import GPy

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    kernels = []
    for i in range(n_fidelities):
        if i == 0:
            kernel = base_kernel_class(input_dim=n_input_dims)
        else:
            kernel = base_kernel_class(input_dim=n_input_dims) * base_kernel_class(input_dim=n_input_dims, ARD=ARD)
        kernels.append(kernel)
    return kernels

if __name__ == "__main__":
    base_kernel_class = GPy.kern.RBF(input_dim=1)
    n_fidelities = 5
    n_input_dims = 1
    ARD = False

    # Sample input values
    X = np.linspace(0, 10, 100).reshape(100, 1)
    fidelities = np.linspace(0, 1, n_fidelities)

    # Create kernels
    kernels = make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD)

    # Print results
    for kernel, fidelity in zip(kernels, fidelities):
        print(f"Fidelity: {fidelity}")
        print(f"Kernel: {kernel}")
        print("---")