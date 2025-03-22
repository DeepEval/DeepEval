import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define input layer with shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer with 16 filters and ReLU activation
    conv = Conv2D(16, (3, 3), activation='relu')(input_layer)

    # Batch normalization
    batch_norm = BatchNormalization()(conv)

    # Global average pooling
    pool = GlobalAveragePooling2D()(batch_norm)

    # Fully connected layers with 128 and 64 units
    dense1 = Dense(128, activation='relu')(pool)
    dense2 = Dense(64, activation='relu')(dense1)

    # Reshape to match initial feature size
    flatten = Flatten()(dense2)

    # Weighted feature maps
    weighted = keras.layers.Multiply()([flatten, input_layer])

    # Concatenate with input layer
    concatenated = Concatenate()([weighted, input_layer])

    # 1x1 convolution and average pooling
    downsampled = Conv2D(32, (1, 1), activation='relu')(concatenated)
    downsampled = GlobalAveragePooling2D()(downsampled)

    # Fully connected layer with 10 units
    output_layer = Dense(10, activation='softmax')(downsampled)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model