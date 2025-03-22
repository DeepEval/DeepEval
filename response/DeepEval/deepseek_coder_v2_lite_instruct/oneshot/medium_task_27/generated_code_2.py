import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional branch
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)

    # Second convolutional branch
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    conv4 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(conv3)

    # Addition of the two branches
    added = Add()([conv2, conv4])

    # Global average pooling
    gap = GlobalAveragePooling2D()(added)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model