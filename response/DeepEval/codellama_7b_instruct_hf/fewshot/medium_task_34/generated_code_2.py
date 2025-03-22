import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, UpSampling2D, Conv2DTranspose
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first part of the model, which extracts deep features
    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D((2, 2))(conv3)

    # Define the second part of the model, which enhances generalization capabilities
    dropout1 = Dropout(0.2)(maxpool3)
    conv4 = Conv2D(256, (3, 3), activation='relu')(dropout1)
    conv5 = Conv2D(256, (3, 3), activation='relu')(conv4)

    # Define the third part of the model, which restores spatial information
    up1 = UpSampling2D((2, 2))(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu')(up1)
    conv7 = Conv2D(128, (3, 3), activation='relu')(conv6)
    up2 = UpSampling2D((2, 2))(conv7)
    conv8 = Conv2D(64, (3, 3), activation='relu')(up2)
    conv9 = Conv2D(64, (3, 3), activation='relu')(conv8)
    up3 = UpSampling2D((2, 2))(conv9)
    conv10 = Conv2D(32, (3, 3), activation='relu')(up3)
    conv11 = Conv2D(32, (3, 3), activation='relu')(conv10)

    # Define the output layer
    flatten = Flatten()(conv11)
    dense = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=dense)

    return model