from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # First feature extraction path: 1x1 convolution
    x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(inputs)

    # Second feature extraction path: 1x1 -> 1x7 -> 7x1 convolutions
    x2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(inputs)
    x2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(x2)
    x2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(x2)

    # Concatenate the outputs of the two paths
    x = Concatenate()([x1, x2])

    # 1x1 convolution to align output dimensions with the input's channel size
    x = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(x)

    # Main path output
    main_output = x

    # Direct branch from input
    branch_output = inputs

    # Merge main path and branch through addition
    x = Add()([main_output, branch_output])

    # Flatten the output for the fully connected layers
    x = Flatten()(x)

    # Fully connected layers for classification
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    # Return the constructed model
    return model