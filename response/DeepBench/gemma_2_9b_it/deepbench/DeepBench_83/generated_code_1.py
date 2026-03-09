import GPy

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
  kernels = []
  for i in range(n_fidelities):
    if i == 0:
      kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
    else:
      previous_kernel = kernels[i-1]
      current_kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
      kernel = previous_kernel * current_kernel + base_kernel_class(input_dim=1, ARD=ARD)
    kernels.append(kernel)
  return kernels

if __name__ == "__main__":
  n_fidelities = 3
  n_input_dims = 2
  kernels = make_non_linear_kernels(GPy.kern.RBF, n_fidelities, n_input_dims)
  print(kernels)