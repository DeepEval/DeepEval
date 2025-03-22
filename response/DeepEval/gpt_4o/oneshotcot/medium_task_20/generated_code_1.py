import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second path: 1x1 convolution followed by two 3x3 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
    
    # Third path: 1x1 convolution followed by a 3x3 convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path3)
    
    # Fourth path: Max pooling followed by a 1x1 convolution
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])
    
    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated)
    
    # Dense layer with 128 units
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer with softmax activation for 10 categories
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model