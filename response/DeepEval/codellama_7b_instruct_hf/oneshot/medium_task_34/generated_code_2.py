import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: feature extraction and classification
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    # Second part: enhance generalization capabilities
    conv4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(maxpool3)
    dropout1 = Dropout(0.2)(conv4)
    conv5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv5)
    conv6 = Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), activation='relu')(dropout2)

    # Third part: upsampling with skip connections
    transconv1 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv6)
    skip1 = keras.layers.concatenate([transconv1, conv5])
    transconv2 = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu')(skip1)
    skip2 = keras.layers.concatenate([transconv2, conv3])
    transconv3 = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu')(skip2)
    skip3 = keras.layers.concatenate([transconv3, conv2])
    transconv4 = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(skip3)
    skip4 = keras.layers.concatenate([transconv4, conv1])
    transconv5 = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(skip4)

    # Output layer
    output_layer = Conv2D(10, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(transconv5)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model