import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Step 2: Add a 1x1 convolutional layer to compress the input channels
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Define a block with parallel convolutions
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)  # 1x1 convolution
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)  # 3x3 convolution
    
    # Step 4: Concatenate the outputs of both paths
    concatenated = Concatenate()([path1, path2])
    
    # Step 5: Flatten the result
    flatten_layer = Flatten()(concatenated)
    
    # Step 6: Add dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model