import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Concatenate, BatchNormalization

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Part 2: Generalization Enhancement
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling and Reconstruction
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = Concatenate()([up6, conv3])
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(merge6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = Concatenate()([up7, conv2])
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(merge7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = Concatenate()([up8, conv1])
    conv8 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(merge8)

    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model