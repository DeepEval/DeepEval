import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Block 1
    def block1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_kernel_initializer='he_uniform')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_kernel_initializer='he_uniform')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor
    
    block1_output = block1(input_tensor=max_pooling)

    # Block 2
    block2_shape = keras.backend.int_shape(block1_output)
    block2_reshaped = Reshape((block2_shape[1], block2_shape[2], block2_shape[3], block2_shape[0]))(block1_output)
    block2_permuted = Permute((3, 1, 2, 4))(block2_reshaped)
    block2_final_reshaped = Reshape((block2_shape[1], block2_shape[2], block2_shape[0], block2_shape[3]))(block2_permuted)

    bath_norm = BatchNormalization()(block2_final_reshaped)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model