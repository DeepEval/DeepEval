import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, DepthwiseConv2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batchnorm_depthwise = BatchNormalization()(depthwise_conv)
    dropout_depthwise = Dropout(0.25)(batchnorm_depthwise)

    # 1x1 convolutional layer for feature extraction
    pointwise_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout_depthwise)
    batchnorm_pointwise = BatchNormalization()(pointwise_conv)
    dropout_pointwise = Dropout(0.25)(batchnorm_pointwise)

    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout_pointwise)

    # Flatten layer
    flatten_layer = Flatten()(max_pooling)

    # Fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model