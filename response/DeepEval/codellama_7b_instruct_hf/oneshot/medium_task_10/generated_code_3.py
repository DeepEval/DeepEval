import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv5)
    conv6 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool5)
    pool6 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv6)
    conv7 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool6)
    pool7 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv7)
    conv8 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool7)
    pool8 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv8)
    conv9 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool8)
    pool9 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv9)
    conv10 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool9)
    pool10 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv10)
    flat = Flatten()(pool10)
    dense = Dense(10, activation='softmax')(flat)
    model = keras.Model(inputs=input_layer, outputs=dense)
    return model