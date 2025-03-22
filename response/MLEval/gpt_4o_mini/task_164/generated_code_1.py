import numpy as np

class Length:
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor

    def compute(self):
        # Length of each capsule is the square root of the sum of squares of its output vectors
        return np.sqrt(np.sum(np.square(self.input_tensor), axis=-1, keepdims=True))

class Mask:
    @staticmethod
    def create_mask(lengths, n_classes, target_class):
        # Creating a mask based on the target class
        mask = np.zeros(lengths.shape)
        mask[np.arange(lengths.shape[0]), target_class] = 1
        return mask

class SquashingFunction:
    @staticmethod
    def squash(vectors):
        # Squashing function to ensure output vectors have a length between 0 and 1
        norm = np.sqrt(np.sum(np.square(vectors), axis=-1, keepdims=True))
        return (norm / (1 + norm**2)) * (vectors / norm)

class CapsuleLayer:
    def __init__(self, n_capsules, n_dimensions):
        self.n_capsules = n_capsules
        self.n_dimensions = n_dimensions

    def forward(self, input_tensor):
        # Assuming input_tensor has the shape (batch_size, n_capsules, n_dimensions)
        # Apply the squashing function
        squashed = SquashingFunction.squash(input_tensor)
        
        # Calculate lengths for each capsule
        lengths = Length(squashed).compute()
        
        return lengths, squashed

def method():
    # Example usage
    batch_size = 10  # Example batch size
    n_capsules = 5   # Number of capsules
    n_dimensions = 16  # Dimensions of each capsule

    # Creating random input tensor
    input_tensor = np.random.rand(batch_size, n_capsules, n_dimensions)

    # Instantiate Capsule Layer
    capsule_layer = CapsuleLayer(n_capsules, n_dimensions)

    # Forward pass
    lengths, squashed_outputs = capsule_layer.forward(input_tensor)

    return lengths, squashed_outputs

# Call the method for validation
output = method()
print("Lengths of capsules:\n", output[0])
print("Squashed outputs:\n", output[1])