# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Reshape, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape for CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Input layer
    inputs = Input(shape=input_shape)

    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # First branch: local feature extraction
    x_branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x_branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_branch1)

    # Second and third branches: downsampling + upsampling
    x_branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(MaxPooling2D(pool_size=(2, 2))(x))
    x_branch2 = UpSampling2D(size=(2, 2))(x_branch2)
    x_branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_branch2)
    x_branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x_branch2))

    x_branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(MaxPooling2D(pool_size=(2, 2))(x))
    x_branch3 = UpSampling2D(size=(2, 2))(x_branch3)
    x_branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_branch3)
    x_branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x_branch3))

    # Concatenate outputs from all branches
    x = Concatenate()([x_branch1, x_branch2, x_branch3])

    # Refine the output using a 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)

    # Flatten the output
    x = Reshape((-1,))(x)

    # Output layer
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model