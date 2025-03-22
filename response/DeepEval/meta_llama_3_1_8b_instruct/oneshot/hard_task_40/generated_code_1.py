import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Reshape, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    flatten1 = Flatten()(avg_pool1)
    
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    flatten2 = Flatten()(avg_pool2)
    
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    flatten3 = Flatten()(avg_pool3)
    
    concatenated_vector = Concatenate()([flatten1, flatten2, flatten3])
    
    reshaped_layer = Reshape((3 * 28 * 28,))(concatenated_vector)
    dense1 = Dense(units=128, activation='relu')(reshaped_layer)
    
    # Second block
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.2)(path1)
        
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Dropout(0.2)(path2)
        
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Dropout(0.2)(path3)
        
        path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Dropout(0.2)(path4)
        
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block_output = block(Reshape((28, 28, 1))(dense1))
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    
    dense2 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model