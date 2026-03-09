import torch
from torch import Tensor

def pad_sequence(sequences: list[Tensor], batch_first: bool = False, padding_value: float = 0.):
    if not isinstance(sequences, Iterable):
        raise RuntimeError("pad_sequence: Expected iterable for input sequences, but got arg of type: %s" % type(sequences))

    # Pad the sequences
    max_length = max(len(seq) for seq in sequences)
    padded_seqs = torch.nn.functional.pad(torch.stack(sequences), (0, max_length), value=padding_value, mode='constant')

    # Determine the output shape based on batch_first
    if batch_first:
        output_shape = (len(sequences), max_length, *padded_seqs.shape[1:])
    else:
        output_shape = (max_length, len(sequences), *padded_seqs.shape[1:])

    return padded_seqs.view(output_shape)

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import itertools

    # Create a list of tensors with different lengths
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5])]

    # Use the pad_sequence function
    output = pad_sequence(sequences, batch_first=False)
    print("Output shape:", output.shape)
    print("Output data:", output.numpy())