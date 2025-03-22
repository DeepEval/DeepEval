import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Batch normalization and ReLU activation
    bn = BatchNormalization()(conv)
    relu = Activation('relu')(bn)

    # Global average pooling
    pool = MaxPooling2D(pool_size=(8, 8), strides=(1, 1), padding='same')(relu)

    # Fully connected layers
    fc1 = Dense(units=64, activation='relu')(pool)
    fc2 = Dense(units=32, activation='relu')(fc1)

    # Reshape output to match initial feature dimensions
    reshape = Reshape((32, 32, 3))(fc2)

    # Multiply output with initial features to generate weighted feature maps
    weighted_maps = Multiply()([reshape, input_layer])

    # Concatenate weighted feature maps with input layer
    concat = Concatenate()([weighted_maps, input_layer])

    # 1x1 convolution and average pooling
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)

    # Flatten output
    flatten = Flatten()(pool1)

    # Fully connected layers
    fc3 = Dense(units=64, activation='relu')(flatten)
    fc4 = Dense(units=10, activation='softmax')(fc3)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc4)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model