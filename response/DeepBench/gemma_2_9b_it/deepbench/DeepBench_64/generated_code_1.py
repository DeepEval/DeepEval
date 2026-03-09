import torch

def rinverse(M, sym=False):
  if M.dim() == 1:
    return 1/M
  elif M.dim() == 2:
    return torch.linalg.inv(M)
  elif M.dim() == 3:
    return torch.einsum('ijk,ikl->jlm', M, torch.linalg.inv(torch.einsum('ijk,jlk->ilk', M, M)))
  else:
    # Blockwise inversion
    batch_size = M.shape[0]
    block_size = int(M.shape[-1] ** 0.5)
    M_blocks = torch.reshape(M, (batch_size, -1, block_size, block_size))
    inv_blocks = torch.zeros_like(M_blocks)
    for i in range(batch_size):
      for j in range(M_blocks.shape[1]):
        inv_blocks[i, j] = torch.linalg.inv(M_blocks[i, j])
    return torch.einsum('bij,bkl->bkl', inv_blocks, M_blocks)

if __name__ == "__main__":
  M = torch.randn(2, 3, 3)
  print(f"Input matrix:\n{M}")
  inv_M = rinverse(M)
  print(f"Inverted matrix:\n{inv_M}")
  print(f"Check: M @ inv_M\n{torch.einsum('ijk,jlm->ilm', M, inv_M)}")