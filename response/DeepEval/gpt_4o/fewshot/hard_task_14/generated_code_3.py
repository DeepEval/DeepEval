import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=32, activation='relu')(global_avg_pool)  # assuming 32 channels
    dense2_main = Dense(units=3, activation='relu')(dense1_main)  # match the input channels
    reshaped_weights = Dense(units=32 * 32 * 3, activation='relu')(dense2_main)
    reshaped_weights = keras.layers.Reshape((32, 32, 3))(reshaped_weights)
    scaled_features = Multiply()([input_layer, reshaped_weights])

    # Branch path
    conv_branch = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine paths
    combined = Add()([scaled_features, conv_branch])

    # Fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model