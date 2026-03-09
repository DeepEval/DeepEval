import torch
from typing import Tuple


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator = None) -> None:
    r"""Fills the input Tensor with values sampled from a Xavier normal distribution.

    The Xavier normal distribution is a probability distribution that is used to initialize the weights of a neural network. It is a good choice for initialization because it helps to ensure that the weights are small and have a similar variance across all layers.

    Args:
        tensor: The input Tensor to be filled with values from the Xavier normal distribution.
        gain: A scaling factor for the distribution. Default is 1.0.
        generator: An optional torch Generator for reproducible results.

    Returns:
        None. The input Tensor is modified in-place.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    a = torch.empty_like(tensor)
    generator = generator if generator is not None else torch.Generator()
    return torch.nn.init.normal_(tensor, 0, std, generator=generator)


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> Tuple[int, int]:
    """Calculates the fan-in and fan-out of a tensor."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan-in and fan-out can only be calculated for tensors with at least 2 dimensions.")

    if dimensions == 2:
        fan_in, fan_out = tensor.size()
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        for s in tensor.size()[2:]:
            receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


if __name__ == "__main__":
    # Example test case: create inputs, call the function, and print results/assert
    tensor = torch.empty(2, 3)
    xavier_normal_(tensor)
    print(tensor)