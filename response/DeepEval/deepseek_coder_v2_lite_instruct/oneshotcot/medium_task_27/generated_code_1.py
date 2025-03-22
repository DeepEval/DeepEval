import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional branch with 3x3 kernel
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv3x3)

    # Second convolutional branch with 5x5 kernel
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(conv5x5)

    # Add the outputs of the two branches
    added = Add()([conv3x3, conv5x5])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(added)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model
    model = Model(inputs=input_layer, outputs=dense2)

    return model