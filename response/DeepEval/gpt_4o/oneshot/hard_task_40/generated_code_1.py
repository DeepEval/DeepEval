import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block with average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)

    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    fc = Dense(units=256, activation='relu')(concatenated)
    
    # Reshape the vector into a 4D tensor
    reshaped = Reshape((4, 4, 16))(fc)  # Adjust dimensions based on output size requirements
    
    # Second block with four parallel paths
    def second_block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.3)(path1)

        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        dropout2 = Dropout(rate=0.3)(path2)
        
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        dropout3 = Dropout(rate=0.3)(path3)

        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        dropout4 = Dropout(rate=0.3)(path4)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3, dropout4])

        return output_tensor

    block_output = second_block(input_tensor=reshaped)

    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model