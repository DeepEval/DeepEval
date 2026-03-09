import numpy as np
import warnings

def hf_state(electrons, orbitals, basis):
    if electrons <= 0:
        raise ValueError("Number of electrons must be greater than zero.")
    if electrons > orbitals:
        raise ValueError("Number of electrons cannot exceed the number of orbitals.")
    
    if basis == "occupation_number":
        hf_state = np.zeros(orbitals)
        hf_state[:electrons] = 1
        return hf_state
    elif basis == "parity":
        hf_state = np.zeros(orbitals)
        hf_state[:electrons] = 1
        hf_state[electrons:] = -1
        return hf_state
    elif basis == "bravyi_kitaev":
        # Bravyi-Kitaev basis mapping
        num_even = orbitals // 2
        num_odd = orbitals - num_even
        hf_state = np.zeros(orbitals)
        hf_state[:num_even//2 + electrons%2] = 1
        if electrons % 2!= 0:
            hf_state[num_even//2 + electrons%2] = 1j
        hf_state[num_even + num_even//2 + electrons%2:] = 1
        if electrons % 2!= 0:
            hf_state[num_even + num_even//2 + electrons%2] = -1j
        return hf_state
    else:
        warnings.warn(f"Ignoring unsupported basis '{basis}'.")

if __name__ == "__main__":
    electrons = 3
    orbitals = 5
    print(hf_state(electrons, orbitals, "occupation_number"))
    print(hf_state(electrons, orbitals, "parity"))
    print(hf_state(electrons, orbitals, "bravyi_kitaev"))