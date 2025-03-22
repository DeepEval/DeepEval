import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)

    # Main path
    pool_main = GlobalAveragePooling2D()(max_pooling)
    dense1_main = Dense(units=128, activation='relu')(pool_main)
    weights_main = Dense(units=3, activation='linear')(dense1_main)
    weights_reshape_main = Reshape((1, 1, 3))(weights_main)
    element_wise_multiply_main = Multiply()([conv, weights_reshape_main])

    # Branch path
    branch_path = conv

    # Combine the outputs from main and branch paths
    combined = Add()([element_wise_multiply_main, branch_path])

    # Apply global average pooling to the combined output
    pool_combined = GlobalAveragePooling2D()(combined)
    dense1 = Dense(units=128, activation='relu')(pool_combined)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model