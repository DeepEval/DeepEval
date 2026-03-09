from typing import List, Tuple, Dict

def _generate_tapes_and_coeffs(tape: QuantumTape, idx: int, atol: float, cache: Dict) -> List[QuantumTape], Tuple[int, int], List[float]:
    if idx in cache:
        return [], (), ()

    # Generate modified tapes
    tapes = []
    for i, op in enumerate(tape.operations):
        if op.name == "Parameter" and op.parameters[idx]:
            tapes.append(tape.copy())
            tapes[-1].operations[i] = op.copy(parameters={idx: op.parameters[idx] - atol})
            tapes[-1].operations[i + 1] = op.copy(parameters={idx: op.parameters[idx] + atol})

    # Compute coefficients
    coeffs = []
    for i, tape in enumerate(tapes):
        coeffs.append(tape.expectation(tape.operations[i + 1]))
        coeffs.append(-tape.expectation(tape.operations[i]))

    # Update cache
    cache[idx] = (tapes, coeffs)

    return tapes, (0, len(tapes) - 1), coeffs

if __name__ == "__main__":
    # Create sample input values
    tape = QuantumTape(QuantumCircuit(1))
    tape.operations = [
        QuantumCircuit.h(0),
        QuantumCircuit.cz(0, 1),
        QuantumCircuit.rz(0, 1),
        QuantumCircuit.rx(0, 1),
    ]
    idx = 1
    atol = 1e-6
    cache = {}

    # Call the function and print the results
    result = _generate_tapes_and_coeffs(tape, idx, atol, cache)
    print(result)