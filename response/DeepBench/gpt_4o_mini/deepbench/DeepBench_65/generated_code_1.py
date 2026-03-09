import numpy as np

def hf_state(electrons, orbitals, basis):
    if electrons <= 0:
        raise ValueError("The number of electrons must be greater than zero.")
    if electrons > orbitals:
        raise ValueError("The number of electrons cannot exceed the number of orbitals.")
    
    if basis == "occupation_number":
        state = np.zeros(2**orbitals)
        for i in range(electrons):
            state[1 << i] = 1
        return state

    elif basis == "parity":
        state = np.zeros(2**orbitals)
        for i in range(electrons):
            state[1 << i] = 1
        parity = (-1)**(electrons % 2)
        return state * parity

    elif basis == "bravyi_kitaev":
        state = np.zeros(2**orbitals)
        for i in range(electrons):
            state[1 << (orbitals - 1 - i)] = 1
        return state

    else:
        raise ValueError("Invalid basis specified. Choose from 'occupation_number', 'parity', or 'bravyi_kitaev'.")

if __name__ == "__main__":
    electrons = 2
    orbitals = 4
    basis = "occupation_number"
    hf_vector = hf_state(electrons, orbitals, basis)
    print(hf_vector)