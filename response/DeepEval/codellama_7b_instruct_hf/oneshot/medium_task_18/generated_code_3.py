import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Extract features at multiple scales using various convolution and pooling operations
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv3)

    # Combine features through concatenation
    concatenated_features = Concatenate()([conv1, conv2, conv3, maxpool])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated_features)
    flattened_features = Flatten()(batch_norm)

    # Output classification results via two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model