import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Concatenate()([x, x, x])

    # Block 2
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Block 3
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Branch path
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(units=128, activation='relu')(branch_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    branch_path = Dense(units=32, activation='relu')(branch_path)

    # Concatenate the outputs from the main path and the branch path
    x = Concatenate()([x, branch_path])

    # Fully connected layer
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model