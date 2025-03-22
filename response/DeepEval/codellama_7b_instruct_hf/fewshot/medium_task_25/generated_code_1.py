import keras
from keras.layers import Input, Conv2D, AvgPool2D, MaxPool2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: single 1x1 convolution
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: average pooling followed by a 1x1 convolution
    avg_pool1 = AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool1)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv3_2)

    # Path 4: 1x1 convolution followed by a 3x3 convolution
    conv4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_1)

    # Multi-scale feature fusion
    fusion_layer = Concatenate()([conv1_1, conv2_1, conv3_3, conv4_2])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(fusion_layer)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    return model