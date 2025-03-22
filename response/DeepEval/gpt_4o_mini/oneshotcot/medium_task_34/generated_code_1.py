import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: Extract features
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Second part: Enhance generalization
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool3)
    dropout1 = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(dropout1)

    # Third part: Upsampling with skip connections
    upsample1 = UpSampling2D(size=(2, 2))(conv5)
    skip1 = Concatenate()([upsample1, conv3])
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip1)

    upsample2 = UpSampling2D(size=(2, 2))(conv6)
    skip2 = Concatenate()([upsample2, conv2])
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip2)

    upsample3 = UpSampling2D(size=(2, 2))(conv7)
    skip3 = Concatenate()([upsample3, conv1])
    conv8 = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(skip3)  # Final output for 10 classes

    model = Model(inputs=input_layer, outputs=conv8)

    return model