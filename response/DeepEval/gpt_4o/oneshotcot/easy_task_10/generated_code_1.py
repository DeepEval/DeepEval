import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolutional layer to increase dimensionality
    conv1x1_increase = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_separable_conv = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1x1_increase)
    
    # Step 4: Add 1x1 convolutional layer to reduce dimensionality
    conv1x1_reduce = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(depthwise_separable_conv)
    
    # Step 5: Flatten the output
    flatten_layer = Flatten()(conv1x1_reduce)
    
    # Step 6: Add dense layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model