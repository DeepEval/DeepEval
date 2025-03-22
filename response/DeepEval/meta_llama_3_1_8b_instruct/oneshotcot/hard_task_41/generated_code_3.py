import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Reshape
from keras.regularizers import l2

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        output_tensor = Concatenate()([path1, path2, path3])

        # Flatten and dropout
        flatten = Flatten()(output_tensor)
        dropout = Dropout(0.2)(flatten)

        return dropout
    
    block1_output = block1(conv)
    flatten = Flatten()(block1_output)
    dense1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(flatten)
    reshape = Reshape((4, 32))(dense1)

    def block2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch7 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        branch8 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch7)

        output_tensor = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch8])

        return output_tensor
        
    block2_output = block2(reshape)
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    dense3 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model