import numpy as np
from qiskit.quantum_info import Operator

def _generate_tapes_and_coeffs(tape: 'QuantumTape', idx: int, atol: float, cache: dict) -> tuple:
  if f"tape_{idx}" in cache:
    return [], [], cache
  modified_tapes = []
  coeffs = []
  start_idx = len(cache["tapes"])
  for i, op in enumerate(tape.operations):
    if isinstance(op, Operator) and op.data.shape[0] == 1 and op.data.shape[1] == 1 and np.isclose(op.data[0,0], idx, atol=atol):
      modified_tapes.append(tape.copy())
      modified_tapes[-1].operations[i] = Operator(np.array([[1.0]])).to_matrix()
      coeffs.append(1.0)
  end_idx = start_idx + len(modified_tapes)
  cache["tapes"].extend(modified_tapes)
  cache[f"tape_{idx}"] = (start_idx, end_idx, coeffs)
  return modified_tapes, (start_idx, end_idx), cache

if __name__ == "__main__":
  tape = QuantumTape()
  tape.append(PauliXGate().to_gate())
  tape.append(Operator(np.array([[1.0, 0.0], [0.0, 1.0]])).to_matrix())
  tape.append(PauliYGate().to_gate())
  idx = 1
  atol = 1e-6
  cache = {"tapes": []}
  modified_tapes, indices, updated_cache = _generate_tapes_and_coeffs(tape, idx, atol, cache)
  print(f"Modified Tapes: {modified_tapes}")
  print(f"Indices: {indices}")
  print(f"Updated Cache: {updated_cache}")