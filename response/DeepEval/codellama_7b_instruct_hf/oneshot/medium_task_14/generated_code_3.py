import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Sequential blocks
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    # Parallel branch
    parallel_conv = Conv2D(64, (3, 3), activation='relu')(input_layer)
    parallel_conv = BatchNormalization()(parallel_conv)
    parallel_conv = ReLU()(parallel_conv)

    # Add outputs from sequential blocks and parallel branch
    merged = keras.layers.concatenate([conv1, conv2, conv3, parallel_conv])

    # Flatten and add fully connected layers
    flattened = Flatten()(merged)
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    return model