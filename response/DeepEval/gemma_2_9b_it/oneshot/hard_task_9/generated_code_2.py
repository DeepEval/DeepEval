import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate branches
    concat_layer = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution for dimension adjustment
    concat_layer = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Fuse with direct connection
    output_layer = keras.layers.Add()([input_layer, concat_layer]) 

    # Flatten and fully connected layers
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=128, activation='relu')(output_layer)
    output_layer = Dense(units=64, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model