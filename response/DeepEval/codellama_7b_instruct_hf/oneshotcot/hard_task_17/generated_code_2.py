import keras
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Conv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    # Block 2
    y = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(64, kernel_size=(3, 3), activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Branch from Block 1
    branch = Dense(128, activation='relu')(x)
    branch = Dense(128, activation='relu')(branch)

    # Fuse Block 2 and Branch
    x = Concatenate()([x, branch])

    # Flatten and add dense layers
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model