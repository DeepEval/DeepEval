import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.regularizers import l2

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def block1(input_tensor):
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        drop1 = Dropout(0.2)(flatten1)
        
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        drop2 = Dropout(0.2)(flatten2)
        
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        drop3 = Dropout(0.2)(flatten3)
        
        output_tensor = Concatenate()([drop1, drop2, drop3])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    reshaped_layer = Dense(units=128, activation='relu')(block1_output)
    reshaped_layer = keras.layers.Reshape((4, 128))(reshaped_layer)
    
    def block2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)
        branch5 = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_tensor)
        branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch5)
        
        output_tensor = Concatenate()([branch1, branch4, branch5])
        return output_tensor
    
    block2_output = block2(reshaped_layer)
    
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model