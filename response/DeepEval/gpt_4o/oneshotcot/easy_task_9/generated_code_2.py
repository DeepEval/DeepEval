import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add a 1x1 convolutional layer to increase dimensionality
    conv1x1_increase = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add a 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_increase)
    
    # Step 4: Add a 1x1 convolutional layer to reduce dimensionality
    conv1x1_reduce = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Step 5: Add the output of the last convolution to the original input layer
    add_layer = Add()([input_layer, conv1x1_reduce])
    
    # Step 6: Add a flatten layer
    flatten_layer = Flatten()(add_layer)
    
    # Step 7: Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model