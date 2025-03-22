import keras
from keras.layers import Input, Conv2D, Add, ZeroPadding2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # 1x1 convolutional layer for channel compression
    channel_compression = Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(input_layer)
    
    # Parallel convolutional paths
    conv1_1x1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(channel_compression)
    conv1_3x3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(channel_compression)
    
    # ReLU activation for each path
    relu1_1x1 = Activation('relu')(conv1_1x1)
    relu1_3x3 = Activation('relu')(conv1_3x3)
    
    # Concatenate the outputs of the two paths
    concat = Concatenate(axis=-1)([relu1_1x1, relu1_3x3])
    
    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate and return the model
model = dl_model()