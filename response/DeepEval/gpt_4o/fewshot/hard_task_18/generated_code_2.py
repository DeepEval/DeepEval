import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Adding input to output of first block
    adding_layer = Add()([input_layer, avg_pool1])

    # Second block
    # Global average pooling to generate channel weights
    global_avg_pool = GlobalAveragePooling2D()(adding_layer)

    # Fully connected layers to refine weights
    fc1 = Dense(units=64, activation='relu')(global_avg_pool)
    fc2 = Dense(units=adding_layer.shape[-1], activation='sigmoid')(fc1)

    # Reshape and multiply with input
    reshaped_weights = keras.layers.Reshape((1, 1, adding_layer.shape[-1]))(fc2)
    weighted_input = Multiply()([adding_layer, reshaped_weights])

    # Flatten and final fully connected layer for classification
    flatten_layer = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model