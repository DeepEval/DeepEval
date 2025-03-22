import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    # Concatenate the outputs of the three branches
    concat_layer = Concatenate()([branch1, branch2, branch3])
    
    # Adjust the output dimensions to match the input image's channel size
    conv_layer = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    
    # Fuse the main path and the branch together through addition
    add_layer = Add()([conv_layer, input_layer])
    
    # Batch normalization
    batch_norm = BatchNormalization()(add_layer)
    
    # Flatten
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model