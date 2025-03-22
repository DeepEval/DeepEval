import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: 1x1 convolution followed by two 3x3 convolutions
    path2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_1)
    path2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_2)
    
    # Path 3: 1x1 convolution followed by a 3x3 convolution
    path3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3_1)
    
    # Path 4: Max pooling followed by a 1x1 convolution
    path4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    path4_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4_1)
    
    # Concatenate outputs from all paths
    concatenated = Concatenate()([path1, path2_3, path3_2, path4_2])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Dense layer with 128 units
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer with 10 units for classification
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model