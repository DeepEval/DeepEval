import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # First extract features through two <convolution, convolution, max pooling> blocks
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Then extract further refined features through two more <convolution, convolution, convolution, max pooling> blocks
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    # Flatten the feature maps and pass them through three fully connected layers to produce the classification results
    flatten = Flatten()(maxpool2)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model