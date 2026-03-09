import torch
import logging
from typing import Optional, Union
from tv_tensors import BoundingBoxes

class BoundingBoxFormat:
    YOLO = 'yolo'
    CXCYWH = 'cxcywh'
    XYWH = 'xywh'
    XYX2Y2 = 'xyx2y2'

def _convert_bounding_box_format(inpt, old_format, new_format):
    if old_format == BoundingBoxFormat.YOLO and new_format == BoundingBoxFormat.XYWH:
        x_center, y_center, width, height = inpt.unbind(dim=-1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        return torch.stack([x1, y1, width, height], dim=-1)
    elif old_format == BoundingBoxFormat.XYWH and new_format == BoundingBoxFormat.YOLO:
        x1, y1, width, height = inpt.unbind(dim=-1)
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        return torch.stack([x_center, y_center, width, height], dim=-1)
    elif old_format == BoundingBoxFormat.CXCYWH and new_format == BoundingBoxFormat.XYWH:
        center_x, center_y, width, height = inpt.unbind(dim=-1)
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        return torch.stack([x1, y1, width, height], dim=-1)
    elif old_format == BoundingBoxFormat.XYWH and new_format == BoundingBoxFormat.CXCYWH:
        x1, y1, width, height = inpt.unbind(dim=-1)
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        return torch.stack([center_x, center_y, width, height], dim=-1)
    elif old_format == BoundingBoxFormat.XYX2Y2 and new_format == BoundingBoxFormat.XYWH:
        x1, y1, x2, y2 = inpt.unbind(dim=-1)
        width = x2 - x1
        height = y2 - y1
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        return torch.stack([x_center, y_center, width, height], dim=-1)
    elif old_format == BoundingBoxFormat.XYWH and new_format == BoundingBoxFormat.XYX2Y2:
        x_center, y_center, width, height = inpt.unbind(dim=-1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        raise ValueError(f"Unsupported format conversion from {old_format} to {new_format}")

def convert_bounding_box_format(inpt, old_format: Optional[str] = None, new_format: Optional[str] = None, inplace: bool = False):
    if new_format is None:
        raise TypeError("new_format cannot be None")
    
    if isinstance(inpt, torch.Tensor):
        if old_format is None:
            raise ValueError("old_format cannot be None for tensor input")
        return _convert_bounding_box_format(inpt, old_format.upper() if isinstance(old_format, str) else old_format, new_format.upper() if isinstance(new_format, str) else new_format)
    elif isinstance(inpt, BoundingBoxes):
        if old_format is not None:
            raise ValueError("old_format should be None for BoundingBoxes input")
        result = _convert_bounding_box_format(inpt.data, inpt.format, new_format.upper() if isinstance(new_format, str) else new_format)
        return BoundingBoxes(result, format=new_format.upper() if isinstance(new_format, str) else new_format)
    else:
        raise TypeError("Input must be a torch.Tensor or tv_tensors.BoundingBoxes")

if __name__ == "__main__":
    # Example usage
    tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    old_format = BoundingBoxFormat.YOLO
    new_format = BoundingBoxFormat.XYWH
    result = convert_bounding_box_format(tensor, old_format, new_format)
    print(result)