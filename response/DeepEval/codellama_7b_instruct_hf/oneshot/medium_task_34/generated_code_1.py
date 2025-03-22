import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10


    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    max_pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(max_pool2)
    max_pool3 = MaxPooling2D((2, 2))(conv3)


    conv4 = Conv2D(64, (3, 3), activation='relu')(max_pool3)
    dropout1 = Dropout(0.2)(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv5)
    conv6 = Conv2D(256, (3, 3), activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(conv6)


    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(dropout3)
    skip1 = Concatenate()([conv4, up1])
    conv7 = Conv2D(64, (3, 3), activation='relu')(skip1)
    up2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    skip2 = Concatenate()([conv2, up2])
    conv8 = Conv2D(64, (3, 3), activation='relu')(skip2)
    up3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    skip3 = Concatenate()([conv1, up3])
    conv9 = Conv2D(32, (3, 3), activation='relu')(skip3)


    output_layer = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)


    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model