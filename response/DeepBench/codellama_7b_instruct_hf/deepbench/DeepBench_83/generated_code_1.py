import GPy

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    kernels = []
    for i in range(n_fidelities):
        kernel = GPy.kern.
        if i > 0:
            kernel = kernel * kernels[i-1]
        if ARD:
            kernel.set_ARD(True)
        kernels.append(kernel)
    return kernels

if __name__ == "__main__":
    X = np.random.rand(10, 2)
    base_kernel_class = kern.Matern
    n_fidelities = 3
    kernels = make_non_linear_kernels(base_kernel_class, n_fidelities, X.shape[1])
    for i, kernel in enumerate(kernels):
        print(f"Kernel {i}: {kernel}")