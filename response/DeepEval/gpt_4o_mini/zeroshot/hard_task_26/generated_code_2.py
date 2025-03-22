import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense, Flatten, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main path
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Branch 2: Max Pooling + 3x3 Convolution + Upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max Pooling + 3x3 Convolution + Upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final convolution in main path
    main_output = Conv2D(32, (1, 1), padding='same', activation='relu')(concatenated)

    # Branch path
    branch_path = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Combine main path and branch path
    combined = Add()([main_output, branch_path])

    # Flatten and classify
    flattened = Flatten()(combined)
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create model
    model = Model(inputs=input_layer, outputs=dense2)

    return model