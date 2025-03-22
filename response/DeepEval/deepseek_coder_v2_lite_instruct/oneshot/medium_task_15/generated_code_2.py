import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Concatenate, Reshape, Multiply, AveragePooling2D, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)

    # Compression using Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(batch_norm1)
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape to match the channels of the initial features
    reshape_layer = Reshape((1, 1, 32))(dense2)

    # Multiply with the initial features to generate weighted feature maps
    weighted_maps = Multiply()([batch_norm1, reshape_layer])

    # Concatenate with the input layer
    concatenated = Concatenate()([input_layer, weighted_maps])

    # Downsample the feature maps using 1x1 convolution and average pooling
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(8, 8))(conv2)
    flatten_layer = Flatten()(avg_pool)

    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model