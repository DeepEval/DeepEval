import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional branch with 3x3 kernel
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_3x3)

    # Second convolutional branch with 5x5 kernel
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(conv_5x5)

    # Add the outputs of the two branches
    added = Add()([conv_3x3, conv_5x5])

    # Global average pooling
    gap = GlobalAveragePooling2D()(added)

    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model