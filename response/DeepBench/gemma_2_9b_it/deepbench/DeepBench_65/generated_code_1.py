import numpy as np

def hf_state(electrons, orbitals, basis):
  if electrons <= 0:
    raise ValueError("Number of electrons must be greater than zero.")
  if electrons > orbitals:
    raise ValueError("Number of electrons cannot exceed the number of orbitals.")
  if basis == "occupation_number":
    state = np.zeros(orbitals)
    state[:electrons] = 1
    return state
  elif basis == "parity":
    if electrons % 2 == 0:
      state = np.array([1, 1])
    else:
      state = np.array([1, -1])
    return state
  elif basis == "bravyi_kitaev":
    if electrons == 1:
      state = np.array([1, 0])
    elif electrons == 2:
      state = np.array([1, 1])
    else:
      raise ValueError("Bravyi-Kitaev basis only defined for 1 and 2 electrons.")
    return state
  else:
    raise ValueError("Invalid basis. Choose from 'occupation_number', 'parity', or 'bravyi_kitaev'.")

if __name__ == "__main__":
  electrons = 3
  orbitals = 5
  basis = "occupation_number"
  state = hf_state(electrons, orbitals, basis)
  print(f"HF state vector in {basis} basis:\n{state}")