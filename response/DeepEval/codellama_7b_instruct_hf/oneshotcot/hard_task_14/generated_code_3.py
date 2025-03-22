import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(input_layer)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Branch path
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(input_layer)
    branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(branch)
    branch = Flatten()(branch)
    branch = Dense(units=128, activation='relu')(branch)
    branch = Dense(units=10, activation='softmax')(branch)

    # Combine main and branch paths
    combined = Add()([x, branch])

    # Final fully connected layers
    combined = Flatten()(combined)
    combined = Dense(units=128, activation='relu')(combined)
    combined = Dense(units=10, activation='softmax')(combined)

    # Create and return model
    model = Model(inputs=input_layer, outputs=combined)
    return model