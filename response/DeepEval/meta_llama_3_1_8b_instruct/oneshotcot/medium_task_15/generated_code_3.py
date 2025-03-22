import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Concatenate, Reshape, Multiply, Conv2DTranspose
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer to extract initial features
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    
    # Compress feature maps using global average pooling and two fully connected layers
    compressed = GlobalAveragePooling2D()(conv)
    dense1 = Dense(units=32, activation='relu')(compressed)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape to match the size of the initial feature
    reshaped = Reshape((1, 1, 32))(dense2)
    
    # Multiply with the initial features to generate weighted feature maps
    weighted = Multiply()([conv, reshaped])
    
    # Concatenate with the input layer
    concatenated = Concatenate()([input_layer, weighted])
    
    # Reduce dimensionality and downsample feature using 1x1 convolution and average pooling
    reduced = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    reduced = BatchNormalization()(reduced)
    reduced = Activation('relu')(reduced)
    reduced = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(reduced)
    
    # Output layer with a single fully connected layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')(reduced)
    output_layer = Activation('softmax')(output_layer)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model