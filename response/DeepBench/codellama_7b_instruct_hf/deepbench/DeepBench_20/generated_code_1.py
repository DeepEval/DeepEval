import torch

def masked_gather(points, idx):
    # Check if the input tensors have matching batch dimensions
    if points.size(0) != idx.size(0):
        raise ValueError("points and idx must have matching batch dimensions")

    # Replace -1 indices with 0 and gather the corresponding points
    idx = idx.masked_fill(-1, 0)
    gathered = points.gather(1, idx)

    # Set the gathered values corresponding to the original -1 indices to 0.0
    gathered = gathered.masked_fill(-1, 0.0)

    return gathered

if __name__ == "__main__":
    # Example usage
    points = torch.tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[13, 14], [15, 16], [17, 18]]
    ])

    idx = torch.tensor([
        [[1, 2], [0, -1], [2, 0]],
        [[0, 2], [-1, 0], [-1, 0]],
        [[2, 0], [0, 2], [-1, 0]]
    ])

    output = masked_gather(points, idx)

    print(output)