import torch
import logging
from tv_tensors import BoundingBoxes

def convert_bounding_box_format(inpt, old_format=None, new_format=None, inplace=False):
    if new_format is None:
        raise TypeError("new_format cannot be None")
    
    if not inplace:
        logging.info('API usage logged')
        
    if isinstance(inpt, torch.Tensor):
        if old_format is None:
            raise ValueError("old_format cannot be None with a tensor input")
        else:
            if old_format == new_format:
                return inpt
            elif old_format == 'tv_tensors.BoundingBoxes':
                old_format = old_format.old_format
            return _convert_bounding_box_format(inpt, old_format, new_format)
    elif isinstance(inpt, BoundingBoxes):
        if old_format is None:
            raise ValueError("old_format cannot be None with a BoundingBoxes object input")
        elif old_format == 'tv_tensors.BoundingBoxes':
            old_format = old_format.old_format
        return BoundingBoxes(_convert_bounding_box_format(inpt.bounding_boxes, old_format, new_format), new_format)
    else:
        raise TypeError("Unsupported input type")


def _convert_bounding_box_format(inpt, old_format, new_format):
    if isinstance(inpt, torch.Tensor):
        return inpt
    elif isinstance(inpt, BoundingBoxes):
        return BoundingBoxes(_convert_bounding_box_format(inpt.bounding_boxes, old_format, new_format), new_format)
    else:
        raise TypeError("Unsupported input type")


if __name__ == "__main__":
    import torch
    import logging
    from tv_tensors import BoundingBoxes

    logging.basicConfig(level=logging.INFO)

    # Sample input values
    inpt_tensor = torch.rand(5, 4)
    inpt_box = BoundingBoxes([(0, 1), (2, 3), (4, 5)], old_format='CORNER_CENTER')

    # Call the function
    converted_tensor = convert_bounding_box_format(inpt_tensor)
    converted_box = convert_bounding_box_format(inpt_box, old_format='CORNER_CENTER', new_format='CORNER_X_CENTER_Y_SIZE')

    # Print the results
    print("Tensor conversion:", converted_tensor)
    print("Box conversion:", converted_box)