import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)

    # Add input to the output of main path
    block1_output = Add()([input_layer, avg_pool])

    # Second Block
    # Global Average Pooling for Squeeze
    global_avg_pool = GlobalAveragePooling2D()(block1_output)

    # Fully Connected layers for Excitation
    fc1 = Dense(units=64, activation='relu')(global_avg_pool)
    fc2 = Dense(units=64, activation='sigmoid')(fc1)

    # Reshape to match the input dimensions
    scale = Reshape((1, 1, 64))(fc2)

    # Multiply the scale with block1 output
    scaled_output = Multiply()([block1_output, scale])

    # Flatten the output
    flatten_layer = Flatten()(scaled_output)

    # Final Dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model