import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2])
        return output_tensor
    
    block_output = block(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block_output)

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv)

    # Matching output dimensions
    matching_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_max_pooling)

    # Summing paths
    summed_output = Add()([bath_norm, matching_conv])

    # Flattening and fully connected layer
    flatten_layer = Flatten()(summed_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model