import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Block 1
    block1 = Conv2D(32, (3, 3), activation='relu')(conv)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)

    # Block 2
    block2 = Conv2D(32, (3, 3), activation='relu')(block1)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)

    # Block 3
    block3 = Conv2D(32, (3, 3), activation='relu')(block2)
    block3 = BatchNormalization()(block3)
    block3 = ReLU()(block3)

    # Add blocks
    added = Add()([conv, block1, block2, block3])

    # Flatten
    flattened = Flatten()(added)

    # Fully connected layers
    dense1 = Dense(64, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model