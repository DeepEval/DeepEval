import keras
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution for dimensionality reduction
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # Branch 3: 1x1 convolution followed by 5x5 convolution
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv3_1)

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_layer)
    conv4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)

    # Concatenate the outputs of these branches
    output_tensor = Concatenate()([conv1, conv2_2, conv3_2, conv4])

    # Flatten the concatenated features
    flatten_layer = Flatten()(output_tensor)

    # Two fully connected layers for feature combination and classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model