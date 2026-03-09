import torch
from torch.utils.data import IterableDataset
from typing import Iterable

def pad_sequence(sequences: Iterable[torch.Tensor], batch_first: bool = False, padding_value: float = 0.0):
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if not isinstance(sequences, Iterable):
            msg = ("pad_sequence: Expected iterable for input sequences, but got arg of type: " 
                   f"{type(sequences)}")
            raise RuntimeError(msg)
    
    lengths = [s.shape[0] for s in sequences]
    max_length = max(lengths)
    
    if batch_first:
        # Create a tensor with the padding value
        padded_sequences = torch.full((len(sequences), max_length) + sequences[0].shape[1:], 
                                      padding_value, dtype=sequences[0].dtype)
        
        # Stack the sequences along the first dimension
        for i, s in enumerate(sequences):
            padded_sequences[i, :s.shape[0]] = s
        
        # Move the sequence dimension to the beginning
        padded_sequences = padded_sequences.flip((0,)).contiguous()
    else:
        # Create a tensor with the padding value
        padded_sequences = torch.full((max_length, len(sequences)) + sequences[0].shape[1:], 
                                      padding_value, dtype=sequences[0].dtype)
        
        # Stack the sequences along the first dimension
        for i, s in enumerate(sequences):
            padded_sequences[:s.shape[0], i] = s
        
    return padded_sequences

if __name__ == "__main__":
    # Create sample input values
    sequences = [torch.randn(3, 5), torch.randn(2, 5), torch.randn(4, 5)]
    
    # Call the function and print the results
    batch_first = False
    padded_sequence = pad_sequence(sequences, batch_first=batch_first)
    print(padded_sequence.shape)
    
    batch_first = True
    padded_sequence = pad_sequence(sequences, batch_first=batch_first)
    print(padded_sequence.shape)