import torch
import tv_tensors
import logging

def convert_bounding_box_format(inpt, old_format, new_format, inplace=False):
    if new_format is None:
        raise TypeError("new_format cannot be None")
    logging.info("API usage: convert_bounding_box_format")
    if isinstance(old_format, str):
        old_format = old_format.upper()
    if isinstance(new_format, str):
        new_format = new_format.upper()
    if isinstance(inpt, torch.Tensor):
        if old_format is not None:
            raise ValueError("old_format cannot be specified for a pure tensor")
        if inplace:
            result = _convert_bounding_box_format(inpt, new_format)
            inpt.data = result.data
            return inpt
        else:
            return _convert_bounding_box_format(inpt, new_format)
    elif isinstance(inpt, tv_tensors.BoundingBoxes):
        if old_format is not None:
            raise ValueError("old_format cannot be specified for a tv_tensors.BoundingBoxes object")
        result = _convert_bounding_box_format(inpt, new_format)
        return tv_tensors.BoundingBoxes(result, new_format)
    else:
        raise TypeError("input must be a pure tensor or a tv_tensors.BoundingBoxes object")

def _convert_bounding_box_format(inpt, new_format):
    # Implement the actual conversion logic here
    # This function takes inpt and new_format as input and returns a torch.Tensor
    pass

if __name__ == "__main__":
    # Sample input values
    inpt = torch.randn(2, 4)
    old_format = "XYWH"
    new_format = "XYXY"

    # Call the function and print the result
    result = convert_bounding_box_format(inpt, old_format, new_format)
    print(result)