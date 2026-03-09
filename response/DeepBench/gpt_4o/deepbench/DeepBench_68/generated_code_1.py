from pennylane import QuantumTape
from pennylane.wires import Wires
import numpy as np

def _generate_tapes_and_coeffs(tape, idx, atol, cache):
    parameter_name = f"param_{idx}"
    
    if parameter_name in cache:
        return [], (0, 0), [], cache

    modified_tapes = []
    coeffs = []

    for op_idx, op in enumerate(tape.operations):
        if idx in op.parameter_deps:
            # Create a new tape with the parameter shifted
            new_params = tape.get_parameters()
            new_params[op.parameter_deps.index(idx)] += atol
            
            new_tape = QuantumTape(
                operations=[op.with_parameters(new_params)],
                measurements=tape.measurements,
                wires=Wires(tape.wires)
            )
            modified_tapes.append(new_tape)
            coeffs.append(1.0 / atol)

    cache[parameter_name] = (modified_tapes, coeffs)
    return modified_tapes, (0, len(modified_tapes)), coeffs, cache

if __name__ == "__main__":
    # Create a sample QuantumTape
    from pennylane import RX, measure

    op1 = RX(0.5, wires=0)
    op2 = RX(0.3, wires=1)
    ops = [op1, op2]

    tape = QuantumTape(
        ops, [measure.expval(op1), measure.var(op2)], wires=[0, 1]
    )
    
    idx = 0  # Index of the trainable parameter
    atol = 1e-2  # Absolute tolerance
    cache = {}  # Empty cache dictionary

    modified_tapes, indices, coeffs, updated_cache = _generate_tapes_and_coeffs(tape, idx, atol, cache)

    print("Modified Tapes:")
    for mt in modified_tapes:
        print(mt)

    print("Indices:", indices)
    print("Coefficients:", coeffs)
    print("Updated Cache:", updated_cache)