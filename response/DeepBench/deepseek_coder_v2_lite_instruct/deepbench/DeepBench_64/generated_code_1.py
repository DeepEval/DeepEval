import torch

def rinverse(M, sym=False):
    if M.dim() == 1:
        return 1 / M
    elif M.dim() == 2:
        return torch.inverse(M)
    else:
        # Blockwise inversion for larger matrices
        n = M.size(-1)
        block_size = 2
        while block_size < n:
            block_size *= 2
        
        # Pad the matrix to make its size a power of 2
        padding = (0, block_size - n) if M.dim() == 3 else (block_size - n,)
        M_padded = torch.nn.functional.pad(M, padding, 'constant', 0)
        
        def block_inv(B):
            return torch.inverse(B[:B.size(0)//2, :B.size(1)//2]) @ \
                   torch.inverse(B[B.size(0)//2:, B.size(1)//2:]) - \
                   torch.inverse(B[:B.size(0)//2, :B.size(1)//2]) @ \
                   B[:B.size(0)//2, B.size(1)//2:] @ \
                   torch.inverse(B[B.size(0)//2:, B.size(1)//2:] + \
                                 B[B.size(0)//2:, :B.size(1)//2] @ \
                                 torch.inverse(B[:B.size(0)//2, :B.size(1)//2]) @ \
                                 B[:B.size(0)//2, B.size(1)//2:])
        
        while M_padded.size(-1) > 2:
            n = M_padded.size(-1)
            half = n // 2
            A11 = M_padded[..., :half, :half]
            A22 = M_padded[..., half:, half:]
            A12 = M_padded[..., :half, half:]
            A21 = M_padded[..., half:, :half]
            M_padded[..., :half, :half] = block_inv(A11)
            M_padded[..., half:, half:] = block_inv(A22)
            M_padded[..., :half, half:] -= block_inv(A11) @ A12 @ block_inv(A22)
            M_padded[..., half:, :half] -= block_inv(A22) @ A21 @ block_inv(A11)
        
        return M_padded

if __name__ == "__main__":
    # Example usage
    batch_size = 3
    matrix_size = 2
    M = torch.randn(batch_size, matrix_size, matrix_size)
    print("Original Matrix M:\n", M)
    inv_M = rinverse(M)
    print("Inverted Matrix inv_M:\n", inv_M)
    identity_check = torch.allclose(torch.matmul(M, inv_M), torch.eye(matrix_size).repeat(batch_size, 1, 1))
    print("Identity check:", identity_check)