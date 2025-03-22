import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Branch 1: 3x3 convolutions
    branch1_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by two 3x3 convolutions
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_conv2)
    
    # Branch 3: Max pooling
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1_conv, branch2_conv3, branch3_pool])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model