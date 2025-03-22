import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer
    depthwise_conv = DepthwiseSeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # 1x1 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Add the output of the 1x1 convolutional layer to the original input layer
    concatenated_output = Concatenate()([input_layer, conv2])
    
    # Batch normalization
    batch_norm = BatchNormalization()(concatenated_output)
    
    # Flatten layer
    flattened_output = Flatten()(batch_norm)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flattened_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model