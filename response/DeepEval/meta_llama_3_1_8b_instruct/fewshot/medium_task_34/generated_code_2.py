import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add, Conv2DTranspose, concatenate
from keras.regularizers import l2

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction through 3 pairs of convolutional and max-pooling layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Enhance generalization capabilities through convolutional, Dropout, and convolutional layers
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    dropout = Dropout(0.2)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    # Upsampling through 3 pairs of convolutional and transposed convolutional layers with skip connections
    up6 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv3])
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up6)
    up7 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv2])
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up7)
    up8 = Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv1])
    conv8 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(up8)

    # Final 1x1 convolutional layer for probability output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv8)
    output_layer = keras.layers.Reshape((10,))(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model