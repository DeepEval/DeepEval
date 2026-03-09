import GPy

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    """
    Creates a list of structured multi-fidelity kernels.
    """
    kernels = []
    for i in range(n_fidelities):
        if i == 0:
            kernels.append(base_kernel_class(input_dim=n_input_dims, ARD=ARD))
        else:
            kernels.append(base_kernel_class(input_dim=n_input_dims, ARD=ARD) * kernels[i - 1] + base_kernel_class(input_dim=n_input_dims, ARD=ARD))
    return kernels

if __name__ == "__main__":
    # Example usage:
    n_fidelities = 3
    n_input_dims = 2
    base_kernel_class = GPy.kern.Matern32
    kernels = make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims)
    print(kernels)