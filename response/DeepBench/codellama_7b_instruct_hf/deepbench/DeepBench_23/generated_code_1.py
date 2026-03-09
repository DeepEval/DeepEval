def pad_sequence(sequences, batch_first=False, padding_value=0):
    # Check that sequences is an iterable
    if not isinstance(sequences, Iterable):
        msg = "pad_sequence: Expected iterable for input sequences, but got arg of type: "
        raise RuntimeError(msg)

    # Initialize variables
    batch_size = len(sequences)
    max_len = 0
    padded_sequences = []

    # Iterate over each sequence in the list
    for seq in sequences:
        # Check that seq is a tensor
        if not isinstance(seq, torch.Tensor):
            msg = "pad_sequence: Expected tensor for sequence, but got arg of type: "
            raise RuntimeError(msg)

        # Get the length of the sequence
        seq_len = seq.size(0)

        # If the sequence is longer than the max length, update the max length
        if seq_len > max_len:
            max_len = seq_len

        # Add the padded sequence to the list
        padded_sequences.append(seq)

    # Create a tensor to store the padded sequences
    padded_tensor = torch.zeros(batch_size, max_len, dtype=torch.float)

    # Iterate over each sequence in the list
    for i, seq in enumerate(padded_sequences):
        # Get the length of the sequence
        seq_len = seq.size(0)

        # Copy the sequence into the padded tensor
        padded_tensor[i, :seq_len] = seq

    # If batch_first is True, transpose the tensor
    if batch_first:
        padded_tensor = padded_tensor.transpose(0, 1)

    return padded_tensor

if __name__ == "__main__":
    # Create a list of sequences
    sequences = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
        torch.tensor([8, 9])
    ]

    # Pad the sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    # Print the padded sequences
    print(padded_sequences)