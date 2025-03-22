import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, MaxPooling2D, Conv2D, BatchNormalization, Concatenate
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    max_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    block1 = Concatenate()([conv1, conv2, conv3, max_pool])

    # Block 2
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(block1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv6)
    block2 = Concatenate()([conv4, conv5, conv6, pool2])

    # Block 3
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(block2)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv9)
    block3 = Concatenate()([conv7, conv8, conv9, pool3])

    # Block 4
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same')(block3)
    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv10)
    conv12 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv11)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv12)
    block4 = Concatenate()([conv10, conv11, conv12, pool4])

    # Block 5
    conv13 = Conv2D(512, (3, 3), activation='relu', padding='same')(block4)
    conv14 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv13)
    conv15 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv14)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv15)
    block5 = Concatenate()([conv13, conv14, conv15, pool5])

    # Flatten and dense layers
    flatten = Flatten()(block5)
    dense1 = Dense(1024, activation='relu')(flatten)
    dense2 = Dense(128, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output)

    return model