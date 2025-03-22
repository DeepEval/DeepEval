import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Conv2DTranspose, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pw_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(dw_conv)
    
    # 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(pw_conv)
    
    # Add the output from the last convolutional layer to the original input layer
    add_layer = Add()([input_layer, conv2])
    
    # Flattening layer
    flatten_layer = Flatten()(add_layer)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model