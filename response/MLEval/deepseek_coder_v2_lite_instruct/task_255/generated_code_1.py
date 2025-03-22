import numpy as np

def method():
    # Example input (batch of logits)
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    
    # Apply softmax to the logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Return the final output
    return softmax_output

# Call the method for validation
output = method()
print(output)