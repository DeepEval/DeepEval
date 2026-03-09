import torch
from typing import Optional
from tvm_tensors import BoundingBoxes

class BoundingBoxFormat:
    XYWH = "XYWH"
    XYXC = "XYXC"

def convert_bounding_box_format(inpt: torch.Tensor, old_format: Optional[BoundingBoxFormat] = None, new_format: Optional[BoundingBoxFormat] = None, inplace: bool = False) -> torch.Tensor:
    if new_format is None:
        raise TypeError("new_format cannot be None")
    if not inplace:
        inpt = inpt.clone()
    if isinstance(old_format, str):
        old_format = old_format.upper()
    if isinstance(new_format, str):
        new_format = new_format.upper()
    if isinstance(inpt, torch.Tensor) or __debug__:
        if old_format is None:
            raise ValueError("old_format cannot be None")
        return _convert_bounding_box_format(inpt, old_format, new_format)
    elif isinstance(inpt, BoundingBoxes):
        if old_format is not None:
            raise ValueError("old_format cannot be specified for tvm_tensors.BoundingBoxes")
        return _convert_bounding_box_format(inpt.tensor, old_format, new_format).to_bounding_boxes(new_format)
    else:
        raise TypeError(f"Input type {type(inpt)} not supported.")



def _convert_bounding_box_format(inpt: torch.Tensor, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat) -> torch.Tensor:
    # This is a placeholder function, you need to implement the actual conversion logic here.
    return inpt

if __name__ == "__main__":
    inpt = torch.tensor([[10, 20, 30, 40]])
    old_format = BoundingBoxFormat.XYWH
    new_format = BoundingBoxFormat.XYXC
    output = convert_bounding_box_format(inpt, old_format, new_format)
    print(output)