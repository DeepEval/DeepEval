import numpy as np
from scipy.sparse import coo_matrix

def compute_adjacency_graph(n_x: int, n_y: int, n_z: int = 1, mask: np.ndarray = None, return_as: type = coo_matrix, dtype: type = int) -> np.ndarray:
  if mask is None:
    mask = np.ones((n_x, n_y, n_z), dtype=bool)
  
  rows = []
  cols = []
  for x in range(n_x):
    for y in range(n_y):
      for z in range(n_z):
        if mask[x, y, z]:
          if x > 0 and mask[x-1, y, z]:
            rows.append(x*n_y*n_z + y*n_z + z)
            cols.append((x-1)*n_y*n_z + y*n_z + z)
          if x < n_x-1 and mask[x+1, y, z]:
            rows.append(x*n_y*n_z + y*n_z + z)
            cols.append((x+1)*n_y*n_z + y*n_z + z)
          if y > 0 and mask[x, y-1, z]:
            rows.append(x*n_y*n_z + y*n_z + z)
            cols.append(x*n_y*n_z + (y-1)*n_z + z)
          if y < n_y-1 and mask[x, y+1, z]:
            rows.append(x*n_y*n_z + y*n_z + z)
            cols.append(x*n_y*n_z + (y+1)*n_z + z)
          if z > 0 and mask[x, y, z-1]:
            rows.append(x*n_y*n_z + y*n_z + z)
            cols.append(x*n_y*n_z + y*n_z + (z-1))
          if z < n_z-1 and mask[x, y, z+1]:
            rows.append(x*n_y*n_z + y*n_z + z)
            cols.append(x*n_y*n_z + y*n_z + (z+1))

  return return_as( (rows, cols), shape=(n_x*n_y*n_z, n_x*n_y*n_z), dtype=dtype)

if __name__ == "__main__":
  n_x = 3
  n_y = 3
  n_z = 2
  mask = np.array([
    [ [True, True, True],
      [True, False, True],
      [True, True, True]],
    [ [True, False, True],
      [False, True, False],
      [True, True, True]]
  ])
  graph = compute_adjacency_graph(n_x, n_y, n_z, mask)
  print(graph.toarray())