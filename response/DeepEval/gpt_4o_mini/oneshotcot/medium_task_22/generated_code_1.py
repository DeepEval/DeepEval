import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, BatchNormalization

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape (32x32 pixels, 3 channels)
    
    # Branch 1: 3x3 Convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Concatenate outputs from all branches
    merged = Concatenate()([branch1, branch2, branch3])
    
    # Batch Normalization
    batch_norm = BatchNormalization()(merged)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model