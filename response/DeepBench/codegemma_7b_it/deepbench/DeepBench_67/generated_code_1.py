import torch

class BoundingBoxFormat:
    # ...

def convert_bounding_box_format(inpt: torch.Tensor, old_format: BoundingBoxFormat = None, new_format: BoundingBoxFormat = None, inplace: bool = False) -> torch.Tensor:
    # ...

    if isinstance(inpt, torch.Tensor):
        # ...
        return _convert_bounding_box_format(inpt, old_format, new_format, inplace)
    elif isinstance(inpt, tv_tensors.BoundingBoxes):
        # ...
        return tv_tensors.BoundingBoxes(
            _convert_bounding_box_format(inpt.tensor, old_format, new_format, inplace), inpt.format
        )
    else:
        raise TypeError(f"Invalid input type {type(inpt)}")

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.randn(1, 4)
    input_bboxes = tv_tensors.BoundingBoxes(torch.randn(1, 4), tv_tensors.BoundingBoxFormat.XYXY)

    # Call the function and print the results
    output_tensor = convert_bounding_box_format(input_tensor)
    output_bboxes = convert_bounding_box_format(input_bboxes, new_format=BoundingBoxFormat.YXYX)

    print("Input Tensor:", input_tensor)
    print("Output Tensor:", output_tensor)
    print("Input Bounding Boxes:", input_bboxes)
    print("Output Bounding Boxes:", output_bboxes)