import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First branch: two sequential 3x3 convolutional layers for local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Second branch: average pooling followed by a 3x3 convolution and a transposed convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2)
    
    # Third branch: similar to second branch
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3)
    
    # Concatenating the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolutional layer to refine the concatenated output
    refined_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Flatten and fully connected layer for classification
    flattened = Flatten()(refined_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model