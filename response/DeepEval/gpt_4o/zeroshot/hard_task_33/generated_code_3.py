from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def build_block(input_tensor, filters):
    # Elevate dimension with 1x1 convolution
    x = Conv2D(filters, (1, 1), activation='relu', padding='same')(input_tensor)
    # Extract features with 3x3 depthwise separable convolution
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same')(x)
    # Reduce dimension with 1x1 convolution
    x = Conv2D(input_tensor.shape[-1], (1, 1), activation='relu', padding='same')(x)
    # Add the block's input
    x = Add()([x, input_tensor])
    return x

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 grayscale
    inputs = Input(shape=input_shape)

    # Define the number of filters for each block
    filters = 32

    # Create three branches
    branch1 = build_block(inputs, filters)
    branch2 = build_block(inputs, filters)
    branch3 = build_block(inputs, filters)

    # Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the output
    flat = Flatten()(concatenated)

    # Fully connected layer to generate classification probabilities
    outputs = Dense(10, activation='softmax')(flat)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Optionally, to visualize the model architecture
model = dl_model()
model.summary()
plot_model(model, to_file='model_architecture.png', show_shapes=True)