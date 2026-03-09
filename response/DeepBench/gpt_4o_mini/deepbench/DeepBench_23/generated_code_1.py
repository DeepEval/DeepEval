import torch
from torch import nn
from collections.abc import Iterable

def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if not isinstance(sequences, Iterable):
            msg = ( "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                    f"{type(sequences)}" )
            raise RuntimeError(msg)
    
    max_length = max(seq.size(0) for seq in sequences)
    padded_sequences = [torch.cat((seq, torch.full((max_length - seq.size(0), *seq.size()[1:]), padding_value))) for seq in sequences]

    stacked = torch.stack(padded_sequences)
    
    if batch_first:
        return stacked.permute(1, 0, *range(2, stacked.dim()))
    else:
        return stacked

if __name__ == "__main__":
    seq1 = torch.tensor([[1], [2], [3]])
    seq2 = torch.tensor([[4], [5]])
    seq3 = torch.tensor([[6]])
    
    sequences = [seq1, seq2, seq3]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)

    print(padded)