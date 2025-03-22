import keras
from keras.layers import Input, Conv2D, Add, Concatenate, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)  # Adjust to match the MNIST data
    input = Input(shape=input_shape)

    # Block 1: Main path and branch path
    def conv_block(x):
        # First convolutional layer
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        branch = x  # This will be part of the branch path
        # Increase feature map dimensions and restore the number of channels
        x = Conv2D(64, (1, 1), activation='relu')(x)
        x = Conv2D(64, (1, 1), activation='relu')(x)
        return x, branch

    # Apply the conv_block to the input
    x, main_branch = conv_block(input)
    
    # Branch path directly connects to the input
    branch = Conv2D(64, (1, 1), activation='relu')(input)

    # Combine outputs from both paths
    x = Add()([x, branch])

    # Block 2: Three max pooling layers
    def pooling_block(x):
        x = MaxPooling2D(pool_size=(1, 1))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        return x

    x = pooling_block(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=input, outputs=output)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])