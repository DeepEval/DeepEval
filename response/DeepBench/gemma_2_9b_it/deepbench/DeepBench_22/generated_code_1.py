import opt_einsum

def einsum(equation, *operands):
    if len(operands) == 0:
        raise ValueError("No operands provided")
    if len(operands) >= 3:
        result = opt_einsum.contract(equation, *operands)
    else:
        result =  sum([sum(a[i] * b[j] for i, j in zip(range(len(a)), range(len(b)))) for a, b in zip(operands, operands[1:])])
    return result

if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    c = [[9, 10], [11, 12]]
    result = einsum('ij,jk->ik', a, b)
    print(result)