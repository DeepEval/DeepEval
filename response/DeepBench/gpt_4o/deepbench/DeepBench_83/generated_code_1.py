import GPy

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    kernels = []

    for fidelity_level in range(n_fidelities):
        if fidelity_level == 0:
            kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
        else:
            prev_kernel = kernels[fidelity_level - 1]
            current_base_kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
            bias_kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
            kernel = current_base_kernel * prev_kernel + bias_kernel
            
        kernels.append(kernel)

    return kernels

if __name__ == "__main__":
    # Example usage
    base_kernel = GPy.kern.RBF
    n_fidelities = 3
    n_input_dims = 2
    ARD = False

    kernels = make_non_linear_kernels(base_kernel, n_fidelities, n_input_dims, ARD)

    # Print kernels
    for i, kernel in enumerate(kernels):
        print(f"Fidelity {i+1}:")
        print(kernel)