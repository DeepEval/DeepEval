import torch
from typing import Iterable

def pad_sequence(sequences, batch_first=False, padding_value=0):
    if not isinstance(sequences, Iterable):
        msg = (
            "pad_sequence: Expected iterable for input sequences, but got arg of type: "
            f"{type(sequences)}"
        )
        raise RuntimeError(msg)

    max_len = max([seq.size(0) for seq in sequences]) if sequences else 0
    out_dim = sequences[0].dim() + 1 if sequences else 0
    out_tensor = sequences[0].new_full((len(sequences), max_len) + sequences[0].shape[1:], padding_value)

    for i, seq in enumerate(sequences):
        length = seq.size(0)
        out_tensor[i, :length, ...] = seq

    if batch_first:
        return out_tensor
    else:
        return out_tensor.transpose(0, 1)

if __name__ == "__main__":
    seq1 = torch.tensor([1, 2, 3])
    seq2 = torch.tensor([4, 5])
    seq3 = torch.tensor([6, 7, 8, 9])

    sequences = [seq1, seq2, seq3]
    padded_seq = pad_sequence(sequences, batch_first=True)
    print(padded_seq)