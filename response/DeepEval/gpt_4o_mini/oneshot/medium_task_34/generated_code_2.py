import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature extraction through 3 pairs of (Conv, MaxPooling)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Part 2: Processing for generalization with (Conv, Dropout, Conv)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(dropout)

    # Part 3: Upsampling with skip connections
    up1 = UpSampling2D(size=(2, 2))(conv5)
    skip1 = Concatenate()([up1, conv3])
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    
    up2 = UpSampling2D(size=(2, 2))(conv6)
    skip2 = Concatenate()([up2, conv2])
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    skip3 = Concatenate()([up3, conv1])
    conv8 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(skip3)

    # Final 1x1 convolutional layer for output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model