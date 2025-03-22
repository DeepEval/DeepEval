import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(64, (3, 3), padding='same')(input_layer)
    x = MaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)

    # Branch path
    branch_input = Conv2D(64, (3, 3), padding='same')(input_layer)
    branch_x = MaxPooling2D()(branch_input)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(x)
    fc2 = Dense(128, activation='relu')(branch_x)

    # Merging paths
    merged_output = keras.layers.add([fc1, fc2])

    # Final output
    output_layer = Dense(10, activation='softmax')(merged_output)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model