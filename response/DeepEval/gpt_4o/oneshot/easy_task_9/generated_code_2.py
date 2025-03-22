import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 Convolutional layer to increase dimensionality
    conv1x1_increase = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 Depthwise separable convolutional layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_increase)
    
    # 1x1 Convolutional layer to reduce dimensionality
    conv1x1_reduce = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Adding the output from the last convolutional layer to the original input layer
    add_layer = Add()([input_layer, conv1x1_reduce])
    
    # Flatten the processed output
    flatten_layer = Flatten()(add_layer)
    
    # Fully connected layer to produce the final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model