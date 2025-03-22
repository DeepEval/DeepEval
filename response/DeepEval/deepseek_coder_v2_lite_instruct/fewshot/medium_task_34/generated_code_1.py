import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Second part: Feature Enhancement
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(pool3)
    drop1 = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(drop1)

    # Third part: Spatial Information Restoration with Skip Connections
    deconv1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2))(conv5)
    concat1 = Add()([deconv1, conv3])
    deconv2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(concat1)
    concat2 = Add()([deconv2, conv2])
    deconv3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2))(concat2)
    concat3 = Add()([deconv3, conv1])

    # Final part: Probability Output
    conv6 = Conv2D(filters=10, kernel_size=(1, 1), activation='sigmoid')(concat3)
    flatten = Flatten()(conv6)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model