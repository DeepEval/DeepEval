import keras
from keras.layers import Input, AveragePooling2D, Dropout, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        dropout1 = Dropout(0.25)(flatten1) 

        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        dropout2 = Dropout(0.25)(flatten2)

        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        dropout3 = Dropout(0.25)(flatten3)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense1 = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense1)

    def block_2(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor
    
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense2 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model