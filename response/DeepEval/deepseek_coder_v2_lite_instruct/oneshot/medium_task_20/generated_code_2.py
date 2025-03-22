import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second path: 1x1 convolution followed by two 3x3 convolutions (which is equivalent to one 5x5 convolution)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Third path: 1x1 convolution followed by one 3x3 convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    
    # Fourth path: Max pooling followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate outputs from all paths
    concatenated = Concatenate()([path1, path2, path3, path4])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model