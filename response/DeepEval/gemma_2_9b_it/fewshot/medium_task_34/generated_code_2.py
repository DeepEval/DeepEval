import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)

    # Part 2: Generalization Enhancement
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(0.5)(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling and Reconstruction
    up1 = UpSampling2D(size=(2, 2))(conv2_2)
    merge1 = concatenate([up1, conv1_3], axis=3)
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(merge1)
    up2 = UpSampling2D(size=(2, 2))(conv3_1)
    merge2 = concatenate([up2, conv1_2], axis=3)
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(merge2)
    up3 = UpSampling2D(size=(2, 2))(conv3_2)
    merge3 = concatenate([up3, conv1_1], axis=3)
    conv3_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(merge3)

    # Final Classification Layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv3_3) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model