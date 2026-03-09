import numpy as np

def hf_state(electrons, orbitals, basis):
    if electrons <= 0:
        raise ValueError("Number of electrons must be greater than zero.")
    if electrons > orbitals:
        raise ValueError("Number of electrons must be less than or equal to the number of orbitals.")

    if basis == "occupation_number":
        state = np.zeros(2**orbitals)
        state[np.sum(np.eye(orbitals, dtype=int)[:electrons])] = 1
        return state

    elif basis == "parity":
        state = np.zeros(2**orbitals)
        for i in range(2**orbitals):
            if bin(i).count('1') == electrons:
                state[i] = 1
        return state

    elif basis == "bravyi_kitaev":
        from qiskit.quantum_info import BravyiKitaevSuperfast
        mapper = BravyiKitaevSuperfast(orbitals)
        state = np.zeros(2**orbitals)
        fermion_op = np.zeros((orbitals, orbitals), dtype=int)
        fermion_op[:electrons, :electrons] = 1
        binary_op = mapper.encode(fermion_op)
        state[binary_op.toarray().reshape(-1)] = 1
        return state

    else:
        raise ValueError("Invalid basis representation. Choose from 'occupation_number', 'parity', or 'bravyi_kitaev'.")

if __name__ == "__main__":
    electrons = 2
    orbitals = 4
    basis = "occupation_number"
    result = hf_state(electrons, orbitals, basis)
    print(result)