import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second path: 1x1 followed by two 3x3 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Third path: 1x1 followed by a single 3x3 convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    
    # Fourth path: max pooling followed by a 1x1 convolution
    path4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate outputs from all paths
    concatenated = Concatenate()([path1, path2, path3, path4])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)
    
    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flattened)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model