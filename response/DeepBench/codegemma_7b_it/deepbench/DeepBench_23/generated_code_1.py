import torch

def pad_sequence(sequences, batch_first=False, padding_value=0):
    batch_sizes = [len(seq) for seq in sequences]
    max_len = max(batch_sizes)
    padded_sequences = []

    for sequence in sequences:
        padded_sequence = sequence + [padding_value] * (max_len - len(sequence))
        padded_sequences.append(padded_sequence)

    if batch_first:
        padded_sequences = torch.tensor(padded_sequences).transpose(0, 1)
    else:
        padded_sequences = torch.tensor(padded_sequences)

    return padded_sequences

if __name__ == "__main__":
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])]
    padded_sequences = pad_sequence(sequences)
    print(padded_sequences)