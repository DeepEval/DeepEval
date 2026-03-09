import torch

def rinverse(M, sym=False):
    if M.dim() < 1:
        raise ValueError("Input must have at least 1 dimension")
    
    if M.dim() == 1:
        # 1D case: invert the scalar elements
        return 1 / M
    
    elif M.dim() == 2:
        # 2D case: Use the formulae for 2x2 matrix inversion
        if M.size(0) != 2 or M.size(1) != 2:
            raise ValueError("For 2D matrices, the size must be 2x2")
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        det = a * d - b * c
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        return torch.tensor([[d, -b], [-c, a]]) / det
    
    elif M.dim() == 3:
        # 3D case: Use torch's inverse function for each matrix in the batch
        if sym:
            return torch.linalg.solve(M, torch.eye(M.size(-1), device=M.device).expand(M.size()))
        else:
            return torch.linalg.inv(M)
    
    else:
        # Blockwise inversion for larger matrices
        q = M.size(-1) // 2
        A = M[..., :q, :q]
        B = M[..., :q, q:]
        C = M[..., q:, :q]
        D = M[..., q:, q:]
        
        A_inv = rinverse(A, sym)
        S = D - C @ A_inv @ B
        S_inv = rinverse(S, sym)
        
        upper_left = A_inv + A_inv @ B @ S_inv @ C @ A_inv
        upper_right = -A_inv @ B @ S_inv
        lower_left = -S_inv @ C @ A_inv
        lower_right = S_inv
        
        top = torch.cat((upper_left, upper_right), dim=-1)
        bottom = torch.cat((lower_left, lower_right), dim=-1)
        return torch.cat((top, bottom), dim=-2)

if __name__ == "__main__":
    # Sample input: 2x2 matrix
    M_2x2 = torch.tensor([[4.0, 7.0], [2.0, 6.0]])
    print("Inverse of 2x2 matrix:\n", rinverse(M_2x2))

    # Sample input: 3x3 matrix
    M_3x3 = torch.tensor([
        [4.0, 7.0, 0.0],
        [2.0, 6.0, 0.0],
        [0.0, 0.0, 5.0]
    ])
    print("Inverse of 3x3 matrix:\n", rinverse(M_3x3))

    # Sample input: 4x4 matrix
    M_4x4 = torch.tensor([
        [4.0, 7.0, 1.0, 0.0],
        [2.0, 6.0, 0.0, 1.0],
        [1.0, 0.0, 5.0, 0.0],
        [0.0, 1.0, 0.0, 3.0]
    ])
    print("Inverse of 4x4 matrix:\n", rinverse(M_4x4))