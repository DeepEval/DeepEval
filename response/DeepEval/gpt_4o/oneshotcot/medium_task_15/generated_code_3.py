import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Multiply, Reshape, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer with batch normalization and ReLU activation
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = Activation('relu')(batch_norm)
    
    # Global average pooling to compress feature maps
    gap = GlobalAveragePooling2D()(relu)
    
    # Two fully connected layers
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape and multiply to generate weighted feature maps
    reshape = Reshape((1, 1, 32))(dense2)
    weighted_features = Multiply()([relu, reshape])
    
    # Concatenate with the initial input layer
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Dimensionality reduction and downsampling using 1x1 convolution and average pooling
    conv_reduced = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    pooled = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_reduced)
    
    # Final fully connected layer for classification
    final_dense = Dense(units=10, activation='softmax')(pooled)
    
    model = keras.Model(inputs=input_layer, outputs=final_dense)
    
    return model