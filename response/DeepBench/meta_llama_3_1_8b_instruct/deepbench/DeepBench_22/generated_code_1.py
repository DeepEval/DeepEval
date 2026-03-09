import numpy as np
import opt_einsum

def einsum(equation, *operands):
    if len(operands) >= 3:
        optimized_equation = opt_einsum.contract_path(*operands, equation)
        return opt_einsum.contract(optimized_equation, *operands)
    else:
        return np.einsum(equation, *operands)

if __name__ == "__main__":
    # Create sample input values
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9]).reshape(3, 1)

    # Call the function
    result1 = einsum('i,j->ij', a, b)
    result2 = einsum('i,j,k->ijk', a, b, c)
    result3 = einsum('i,j->ij', a, b).reshape(-1)
    result4 = einsum('...->...', a[:, None, None])
    result5 = einsum('i,j->ij', a, b)

    # Print the results
    print(result1)
    print(result2)
    print(result3)
    print(result4)
    print(result5)