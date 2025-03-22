import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Part 2: Generalization Enhancement
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(rate=0.25)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling and Reconstruction
    up6 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(merge6)
    
    up7 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(merge7)
    
    up8 = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, input_layer], axis=3)
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(merge8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model