import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    avg_pooling1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pooling2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pooling3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    flatten_layer = Flatten()(avg_pooling3)
    concat_layer = Concatenate()([flatten_layer, Flatten()(avg_pooling2), Flatten()(avg_pooling1)])
    dense_layer = Dense(units=128, activation='relu')(concat_layer)
    reshape_layer = Reshape((128, 1, 1))(dense_layer)
    
    # Second Block
    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(path4)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        output_tensor = Dropout(0.2)(output_tensor)

        return output_tensor
    
    block_output = block(reshape_layer)
    batch_norm_layer = BatchNormalization()(block_output)
    flatten_layer = Flatten()(batch_norm_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model