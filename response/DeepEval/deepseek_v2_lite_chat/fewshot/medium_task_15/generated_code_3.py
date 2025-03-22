import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, GlobalAveragePooling2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer with batch normalization and ReLU activation
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # Average Pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv1)

    # Flatten the output and add two fully connected layers
    flatten = GlobalAveragePooling2D()(avg_pool)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape the dense layer output to match the input feature size
    reshape = Reshape((32, 32, 3))(dense2)

    # Concatenate reshaped dense layer output with initial feature map
    concat_layer = keras.layers.concatenate([reshape, conv1])

    # Downsample the feature map using 1x1 convolution and average pooling
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(concat_layer)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    avg_pool_down = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(avg_pool_down)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model