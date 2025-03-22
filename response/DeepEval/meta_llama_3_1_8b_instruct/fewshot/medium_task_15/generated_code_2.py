import keras
from keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Extract initial features using convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Batch normalization and ReLU activation
    batch_norm = BatchNormalization()(conv)
    relu = keras.layers.Activation('relu')(batch_norm)
    
    # Global average pooling to compress feature maps
    global_avg_pool = GlobalAveragePooling2D()(relu)
    
    # Two fully connected layers to adjust dimensions
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape to match the size of initial feature
    reshaped = Reshape(target_shape=(1, 1, 32))(dense2)
    
    # Multiply with initial features to generate weighted feature maps
    multiply = Multiply()([reshaped, conv])
    
    # Concatenate with the input layer
    concatenated = Concatenate()([multiply, input_layer])
    
    # Reduce dimensionality and downsample using 1x1 convolution and average pooling
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(avg_pool)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model