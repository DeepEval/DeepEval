import torch
from torchvision.ops.boxes import convert_boxes_to_coco_format
from torch import Tensor
from torchvision import ops
import logging
import tv_tensors
from typing import Optional, Union

class BoundingBoxFormat:
    def __init__(self, name):
        self.name = name

def _convert_bounding_box_format(tensor: Tensor, old_format: str, new_format: str, inplace: bool = False) -> Tensor:
    if old_format.lower() == 'coco' and new_format.lower() == 'xyxy':
        return ops.convert_boxes_to_coco_format(tensor, inplace=inplace)
    elif old_format.lower() == 'xyxy' and new_format.lower() == 'coco':
        return ops.convert_boxes_to_coco_format(tensor, inplace=inplace)
    else:
        raise ValueError('Unsupported format conversion.')

def convert_bounding_box_format(tensor: Union[Tensor, tv_tensors.BoundingBoxes], 
                                old_format: Optional[str] = None, 
                                new_format: Optional[str] = None, 
                                inplace: bool = False) -> Union[Tensor, tv_tensors.BoundingBoxes]:
    if new_format is None:
        raise TypeError('new_format cannot be None')
    
    logging.info('API usage: convert_bounding_box_format(tensor, old_format, new_format, inplace)')
    
    if isinstance(tensor, Tensor):
        if not old_format:
            raise ValueError('old_format cannot be None')
        old_format = str(old_format).upper() if isinstance(old_format, str) else old_format
        new_format = str(new_format).upper() if isinstance(new_format, str) else new_format
        return _convert_bounding_box_format(tensor, old_format, new_format, inplace=inplace)
    elif isinstance(tensor, tv_tensors.BoundingBoxes):
        if old_format:
            raise ValueError('old_format cannot be None for tv_tensors.BoundingBoxes')
        old_format = str(old_format).upper() if isinstance(old_format, str) else old_format
        new_format = str(new_format).upper() if isinstance(new_format, str) else new_format
        result = _convert_bounding_box_format(tensor.boxes, old_format, new_format, inplace=inplace)
        return tv_tensors.BoundingBoxes(tensor.labels, result)
    else:
        raise TypeError('Input must be a torch.Tensor or a tv_tensors.BoundingBoxes object')

if __name__ == "__main__":
    tensor = torch.tensor([[10, 10, 20, 20]])
    new_tensor = convert_bounding_box_format(tensor, 'xyxy', 'coco')
    print(new_tensor)
    
    bboxes = tv_tensors.BoundingBoxes([0], tensor)
    new_bboxes = convert_bounding_box_format(bboxes, old_format='coco', new_format='xyxy')
    print(new_bboxes)