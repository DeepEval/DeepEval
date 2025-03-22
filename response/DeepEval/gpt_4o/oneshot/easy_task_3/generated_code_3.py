import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: <convolution, convolution, max pooling>
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    # Second block: <convolution, convolution, max pooling>
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(max_pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    # Third block: <convolution, convolution, convolution, max pooling>
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(max_pool2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3_2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)

    # Fourth block: <convolution, convolution, convolution, max pooling>
    conv4_1 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(max_pool3)
    conv4_2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4_2)
    max_pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)

    # Flatten and fully connected layers
    flatten = Flatten()(max_pool4)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model