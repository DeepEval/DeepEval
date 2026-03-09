import torch
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from collections.abc import Iterable

def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()): 
        if not isinstance(sequences, Iterable):
            msg = (
                "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                f"{type(sequences)}"
            )
            raise RuntimeError(msg)
    return torch_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

if __name__ == "__main__":
    seq1 = torch.tensor([1, 2, 3])
    seq2 = torch.tensor([4, 5])
    seq3 = torch.tensor([6])

    sequences = [seq1, seq2, seq3]

    # Using pad_sequence with batch_first=False (default)
    padded_sequences = pad_sequence(sequences)
    print("Padded sequences with batch_first=False:")
    print(padded_sequences)

    # Using pad_sequence with batch_first=True
    padded_sequences_batch_first = pad_sequence(sequences, batch_first=True)
    print("Padded sequences with batch_first=True:")
    print(padded_sequences_batch_first)