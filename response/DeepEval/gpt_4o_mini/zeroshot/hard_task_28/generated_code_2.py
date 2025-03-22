import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LayerNormalization, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = Input(shape=input_shape)

    # Main path
    # 1. Depthwise convolution (7x7)
    main_path = Conv2D(filters=3, kernel_size=(7, 7), padding='same', groups=3)(inputs)
    
    # 2. Layer normalization
    main_path = LayerNormalization()(main_path)

    # 3. Two sequential 1x1 pointwise convolution layers
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path (direct connection to input)
    branch_path = inputs

    # Combine main path and branch path
    combined = Add()([main_path, branch_path])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=inputs, outputs=dense2)

    return model

# Example of using the function to create the model
model = dl_model()
model.summary()