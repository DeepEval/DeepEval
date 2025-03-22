import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, Conv2DTranspose
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First stage of convolution and max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second stage of convolution and max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    dropout1 = Dropout(0.5)(conv3)

    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(conv4)

    # Upsampling and skip connections
    upsample1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(dropout2)
    concat1 = Concatenate()([upsample1, conv2])
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)

    upsample2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    concat2 = Concatenate()([upsample2, conv1])
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv6)

    # 1x1 convolutional layer for final classification
    conv7 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv6)
    flatten_layer = Flatten()(conv7)

    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model