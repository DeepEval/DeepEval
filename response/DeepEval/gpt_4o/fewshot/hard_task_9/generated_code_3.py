import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by a 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    # Concatenating the outputs from the three branches
    concat_layer = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolution to adjust the output dimensions
    adjusted_layer = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    
    # Main path and direct input branch fused through addition
    adding_layer = Add()([adjusted_layer, input_layer])
    
    # Final classification through fully connected layers
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model