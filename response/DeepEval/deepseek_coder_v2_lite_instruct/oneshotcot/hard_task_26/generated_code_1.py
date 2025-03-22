import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch 2
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(main_path)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(branch2)

    # Branch 3
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(main_path)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(branch3)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolutional layer for the main path
    main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the main path and branch path
    added = Add()([main_output, branch_path])

    # Flatten the output
    flattened = Flatten()(added)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()