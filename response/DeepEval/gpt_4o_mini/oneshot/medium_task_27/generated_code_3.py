import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1: Convolutional Layer with 3x3 kernel
    branch1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: Convolutional Layer with 5x5 kernel
    branch2_conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of both branches through addition
    combined = Add()([branch1_conv, branch2_conv])

    # Apply Global Average Pooling
    pooled_output = GlobalAveragePooling2D()(combined)

    # Create two fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled_output)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer for class probabilities (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model