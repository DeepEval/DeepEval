import GPy

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    kernels = []
    for i in range(n_fidelities):
        if i == 0:
            kernels.append(base_kernel_class(input_dim=n_input_dims, ARD=ARD))
        else:
            kernels.append(kernels[i-1] * base_kernel_class(input_dim=n_input_dims, ARD=ARD) + base_kernel_class(input_dim=n_input_dims, ARD=ARD))
    return kernels

if __name__ == "__main__":
    base_kernel = GPy.kern.RBF
    n_fidelities = 3
    n_input_dims = 2
    kernels = make_non_linear_kernels(base_kernel, n_fidelities, n_input_dims, ARD=True)
    for kernel in kernels:
        print(kernel)