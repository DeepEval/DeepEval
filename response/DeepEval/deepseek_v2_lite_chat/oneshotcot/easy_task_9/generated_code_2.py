import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    # 1x1 convolutional layer to reduce dimensionality while maintaining stride
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Add the output of the first 1x1 layer back to the original input
    added_layer = Concatenate()([input_layer, conv2])
    
    # MaxPooling2D for feature extraction
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(added_layer)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(max_pooling)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model