import tensorlayer as tl

def _J(theta: tl.TensorType) -> tl.TensorType:
    order = theta.order
    if order == 0:
        return tl.ones_like(theta)
    elif order == 1:
        return -theta
    elif order == 2:
        return tl.zeros_like(theta)
    else:
        raise ValueError("Order must be 0, 1, or 2")

if __name__ == "__main__":
    # Create a sample input tensor with order 1
    theta = tl.random.normal((2, 2))
    theta.order = 1
    
    # Call the function
    result = _J(theta)
    
    # Print the result
    print("Result:", result)