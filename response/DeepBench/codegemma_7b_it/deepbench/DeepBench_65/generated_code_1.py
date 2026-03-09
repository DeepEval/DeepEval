import numpy as np

def hf_state(electrons, orbitals, basis):
    if electrons <= 0:
        raise ValueError("Number of electrons must be greater than zero.")

    if electrons > orbitals:
        raise ValueError("Number of electrons cannot exceed the number of orbitals.")

    # Implementation of HF state vector generation logic based on the provided basis.
    # ...

    return hf_vector

if __name__ == "__main__":
    # Example usage:
    electrons = 2
    orbitals = 4
    basis = "occupation_number"

    hf_vector = hf_state(electrons, orbitals, basis)

    print(hf_vector)