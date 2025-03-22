import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First sequential block
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Second sequential block
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Third sequential block
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Parallel branch
    parallel_conv = Conv2D(128, (3, 3), activation='relu')(pool3)

    # Add the outputs from all paths
    merged = Concatenate()([pool1, pool2, pool3, parallel_conv])

    # Flatten and pass through two fully connected layers
    flattened = Flatten()(merged)
    dense1 = Dense(128, activation='relu')(flattened)
    output = Dense(10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model