import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def branch(input_tensor, kernel_size):
        if kernel_size == (1, 1):
            conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        elif kernel_size == (3, 3):
            conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        elif kernel_size == (5, 5):
            conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        else:
            pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        return conv or pool 

    branch1 = branch(input_layer, (1, 1))
    branch2 = branch(input_layer, (3, 3))
    branch3 = branch(input_layer, (5, 5))
    branch4 = branch(input_layer, (3, 3))

    concat_layer = Concatenate()( [branch1, branch2, branch3, branch4] )

    block2_input = GlobalAveragePooling2D()(concat_layer)
    dense1 = Dense(units=64, activation='relu')(block2_input)
    dense2 = Dense(units=32, activation='relu')(dense1)

    weights = Reshape((32, 32, 64))(dense2)  
    element_wise_mul = keras.layers.Multiply()([concat_layer, weights])
    
    output_layer = Flatten()(element_wise_mul)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model