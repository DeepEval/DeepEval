import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply
from keras.layers import GlobalAveragePooling2D as GAP

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
        
    block_output = block(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block_output)
    
    # Global Average Pooling
    gap = GAP()(bath_norm)

    # Reshape to 1D
    reshape = Reshape((64,))(gap)

    # Dense Layer 1
    dense1 = Dense(units=128, activation='relu')(reshape)

    # Dense Layer 2
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape to match the input's shape
    reshape2 = Reshape((3, 3, 64))(gap)

    # Element-wise multiplication
    mul = Multiply()([reshape2, reshape])

    # Flatten the result
    flatten_layer = Flatten()(mul)

    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model