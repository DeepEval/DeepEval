import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 color channel (grayscale)

    # Function to create a branch
    def create_branch(input_layer):
        # Depthwise separable convolution
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        x = Dropout(0.25)(x)  # Dropout to mitigate overfitting
        
        # 1x1 convolution to extract features
        x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        x = Dropout(0.25)(x)  # Dropout to mitigate overfitting
        
        return x
    
    # Input layer
    inputs = Input(shape=input_shape)

    # Create three branches
    branch1 = create_branch(inputs)
    branch2 = create_branch(inputs)
    branch3 = create_branch(inputs)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    x = Flatten()(concatenated)
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Output layer for 10 classes (MNIST has 10 classes: digits 0-9)
    outputs = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
# Print model summary
model.summary()