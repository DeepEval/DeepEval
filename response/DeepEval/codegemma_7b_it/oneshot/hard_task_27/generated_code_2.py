import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, LeakyReLU, DepthwiseConv2D, SeparableConv2D, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Depthwise separable convolutional layer
    conv1 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', use_bias=False)(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    # Average pooling
    avg_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

    # Channel-wise feature transformation
    conv2 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='valid', use_bias=False)(avg_pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    # Combine original input with processed features
    concat = Concatenate()([input_layer, conv3])

    # Addition operation
    conv4 = Conv2D(filters=128, kernel_size=(1, 1), padding='valid', use_bias=False)(concat)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)

    # Global average pooling
    avg_pool2 = Lambda(lambda x: K.mean(x, axis=(1, 2)))(conv4)

    # Final fully connected layers
    dense1 = Dense(units=128, activation='relu')(avg_pool2)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model