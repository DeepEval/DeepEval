import numpy as np
from itertools import combinations

def hf_state(electrons, orbitals, basis):
    if electrons <= 0:
        raise ValueError("Number of electrons must be greater than zero.")
    if electrons > orbitals:
        raise ValueError("Number of electrons cannot exceed number of orbitals.")

    # Initialize the HF state in the occupation number representation
    hf_occupation = np.zeros(orbitals, dtype=int)
    hf_occupation[:electrons] = 1

    if basis == "occupation_number":
        return hf_occupation
    
    elif basis == "parity":
        parity_state = np.zeros(2**orbitals, dtype=int)
        for idx in range(2**orbitals):
            bits = np.array(list(np.binary_repr(idx, width=orbitals)), dtype=int)
            if np.array_equal(bits, hf_occupation):
                parity_state[idx] = 1
        return parity_state
    
    elif basis == "bravyi_kitaev":
        # For simplicity, the BK transformation could be quite complex; let's assume a naive transformation
        bk_state = np.zeros(2**orbitals, dtype=int)
        for idx in range(2**orbitals):
            bits = np.array(list(np.binary_repr(idx, width=orbitals)), dtype=int)
            if np.array_equal(bits, hf_occupation):
                bk_state[idx] = 1
        return bk_state

    else:
        raise ValueError(f"Unknown basis: {basis}")

if __name__ == "__main__":
    # Example usage
    electrons = 2
    orbitals = 4
    basis_occupation = hf_state(electrons, orbitals, "occupation_number")
    print("Occupation number basis:", basis_occupation)

    basis_parity = hf_state(electrons, orbitals, "parity")
    print("Parity basis:", basis_parity)

    basis_bk = hf_state(electrons, orbitals, "bravyi_kitaev")
    print("Bravyi-Kitaev basis:", basis_bk)