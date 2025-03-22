import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Initial feature extraction
    conv_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv_layer)
    relu = ReLU()(batch_norm)

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)

    # Two fully connected layers to adjust dimensions
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape to match the number of channels of the initial features
    reshaped = Reshape((1, 1, 64))(dense2)  # Reshape to (1, 1, channels)

    # Multiply to generate weighted feature maps
    weighted_features = Multiply()([relu, reshaped])

    # Concatenate with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])

    # Downsampling using 1x1 convolution and average pooling
    conv1x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv1x1)

    # Final fully connected layer for classification
    flatten = GlobalAveragePooling2D()(avg_pool)  # Flattening the output
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model