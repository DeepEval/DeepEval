import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)
    block1_output = Add()([input_layer, avg_pool1])  # Combining input and main path output

    # Second block with SE-like attention mechanism
    global_avg_pool = GlobalAveragePooling2D()(block1_output)
    fc1 = Dense(units=32, activation='relu')(global_avg_pool)
    fc2 = Dense(units=32, activation='sigmoid')(fc1)
    scale_weights = Reshape((1, 1, 32))(fc2)
    scaled_output = Multiply()([block1_output, scale_weights])  # Element-wise multiplication

    # Flatten and Dense layer for final classification
    flatten_layer = Flatten()(scaled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model