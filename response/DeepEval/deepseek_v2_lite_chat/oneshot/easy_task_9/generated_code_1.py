import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    batch_norm1 = BatchNormalization()(depthwise_conv)
    
    # 1x1 convolutional layer for dimensionality reduction
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    
    # Add the outputs of the 1x1 convolutional layer to the original input layer
    add = Add()([conv1, conv2])
    
    # Flatten and Fully Connected Layers
    flatten = Flatten()(add)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model