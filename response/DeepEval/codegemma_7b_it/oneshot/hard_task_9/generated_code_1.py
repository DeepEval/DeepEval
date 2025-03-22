import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1x1)
    
    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    conv3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1x1)
    conv3_3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_3x3_1)
    
    # Concatenate outputs from three branches
    concat = Concatenate()([conv1_1x1, conv2_3x3, conv3_3x3_2])
    
    # 1x1 convolution to adjust output dimensions
    conv_final = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Branch directly connecting to input
    branch_direct = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main path and branch
    fused = keras.layers.add([conv_final, branch_direct])
    
    # Classification layers
    flatten = Flatten()(fused)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()