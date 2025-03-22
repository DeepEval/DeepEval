import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(conv1)
    
    # Step 4: Add 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(depthwise_conv)
    
    # Step 5: Flatten the output
    flatten_layer = Flatten()(conv2)
    
    # Step 6: Add dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model