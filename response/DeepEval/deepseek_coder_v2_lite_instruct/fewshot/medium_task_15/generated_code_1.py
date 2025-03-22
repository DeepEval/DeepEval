import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Concatenate, Multiply, AveragePooling2D, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(relu1)

    # Two fully connected layers
    fc1 = Dense(units=64, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Reshape to match the channels of initial features
    reshaped = keras.layers.Reshape((1, 1, 64))(fc2)

    # Multiply with initial features to generate weighted feature maps
    weighted_maps = Multiply()([relu1, reshaped])

    # Concatenate weighted feature maps with the input layer
    concatenated = Concatenate()([input_layer, weighted_maps])

    # 1x1 convolution and average pooling to reduce dimensionality and downsample
    conv_pool = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_pool)

    # Flatten and final fully connected layer
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model