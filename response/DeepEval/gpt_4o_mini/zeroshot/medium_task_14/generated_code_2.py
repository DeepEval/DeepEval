import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    block1_conv = Conv2D(32, (3, 3), padding='same')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_relu = ReLU()(block1_bn)

    # Second Block
    block2_conv = Conv2D(64, (3, 3), padding='same')(block1_relu)
    block2_bn = BatchNormalization()(block2_conv)
    block2_relu = ReLU()(block2_bn)

    # Third Block
    block3_conv = Conv2D(128, (3, 3), padding='same')(block2_relu)
    block3_bn = BatchNormalization()(block3_conv)
    block3_relu = ReLU()(block3_bn)

    # Parallel branch for direct convolution
    parallel_conv = Conv2D(64, (3, 3), padding='same')(input_layer)
    parallel_bn = BatchNormalization()(parallel_conv)
    parallel_relu = ReLU()(parallel_bn)

    # Merge all paths
    merged = Add()([block3_relu, parallel_relu])

    # Flatten the output
    flattened = Flatten()(merged)

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model

# Example of using the function
model = dl_model()
model.summary()  # Display the model summary