import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    global_pooling = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=32, activation='relu')(global_pooling)
    dense2_main = Dense(units=3, activation='sigmoid')(dense1_main)  # Assuming input channels are 3
    reshaped_weights = keras.layers.Reshape((1, 1, 3))(dense2_main)
    scaled_input = Multiply()([input_layer, reshaped_weights])

    # Branch path
    conv_branch = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine paths
    combined_output = Add()([scaled_input, conv_branch])

    # Fully connected layers for final classification
    flatten_layer = Flatten()(combined_output)
    dense1_fc = Dense(units=256, activation='relu')(flatten_layer)
    dense2_fc = Dense(units=128, activation='relu')(dense1_fc)
    output_layer = Dense(units=10, activation='softmax')(dense2_fc)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model