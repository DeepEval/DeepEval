import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    avg_pool = GlobalAveragePooling2D()(input_layer)
    flatten_avg_pool = Flatten()(avg_pool)
    dense1 = Dense(units=input_layer.shape[-1], activation='relu')(flatten_avg_pool)
    dense2 = Dense(units=input_layer.shape[-1], activation='relu')(dense1)
    reshape_dense2 = Reshape((1, 1, input_layer.shape[-1]))(dense2)
    multiply_layer = Multiply()([reshape_dense2, input_layer])

    # Branch Path
    conv_branch = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), padding='same')(input_layer)

    # Combine paths
    add_layer = Add()([multiply_layer, conv_branch])

    # Classification layers
    flatten_add_layer = Flatten()(add_layer)
    dense3 = Dense(units=512, activation='relu')(flatten_add_layer)
    dense4 = Dense(units=256, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model