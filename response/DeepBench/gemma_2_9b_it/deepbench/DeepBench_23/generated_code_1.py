import torch

def pad_sequence(sequences, batch_first=False, padding_value=0.):
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if not isinstance(sequences, Iterable):
            msg = ( "pad_sequence: Expected iterable for input sequences, but got arg of type: " f"{type(sequences)}" )
            raise RuntimeError(msg)
    
    max_len = max(len(seq) for seq in sequences)
    
    if batch_first:
      padded_sequences = torch.zeros(len(sequences), max_len, *sequences[0].size()[1:], dtype=sequences[0].dtype, device=sequences[0].device)
    else:
      padded_sequences = torch.zeros(max_len, len(sequences), *sequences[0].size()[1:], dtype=sequences[0].dtype, device=sequences[0].device)
    
    for i, seq in enumerate(sequences):
      end = min(max_len, len(seq))
      padded_sequences[i[:end], :end, :] = seq[:end, :]

    return padded_sequences.fill_(padding_value)



if __name__ == "__main__":
    sequences = [torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32), torch.tensor([[1,2],[3,4,5]], dtype=torch.float32), torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float32)]
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.)
    print(padded_sequences)