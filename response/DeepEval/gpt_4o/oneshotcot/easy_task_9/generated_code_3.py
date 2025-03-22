import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 1: 1x1 Convolution to increase dimensionality
    conv1x1_increase = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: 3x3 Depthwise Separable Convolution for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_increase)
    
    # Step 3: 1x1 Convolution to reduce dimensionality
    conv1x1_reduce = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Step 4: Add the processed output to the original input
    added = Add()([input_layer, conv1x1_reduce])
    
    # Step 5: Flatten the result
    flatten_layer = Flatten()(added)
    
    # Step 6: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model