from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional branch 1: 1x1 Convolution
    conv_1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Convolutional branch 2: 3x3 Convolution
    conv_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Convolutional branch 3: 5x5 Convolution
    conv_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)

    # Max Pooling branch: 3x3 Max Pooling
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate all branches
    concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5, max_pool])

    # Flatten the concatenated features
    flatten = Flatten()(concatenated)

    # Fully connected layer 1
    fc1 = Dense(128, activation='relu')(flatten)

    # Fully connected layer 2
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model