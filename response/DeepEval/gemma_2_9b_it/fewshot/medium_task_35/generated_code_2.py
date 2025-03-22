import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    # Stage 2: Feature Extraction and Dropout
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3_1)
    dropout = Dropout(0.25)(conv3_2)

    # Stage 3: Upsampling and Skip Connections
    up1 = UpSampling2D(size=(2, 2))(dropout)
    merge1 = Concatenate()([up1, conv2_2])
    conv4_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(merge1)
    conv4_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv4_1)

    up2 = UpSampling2D(size=(2, 2))(conv4_2)
    merge2 = Concatenate()([up2, conv1_2])
    conv5_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(merge2)
    conv5_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv5_1)

    # Final Output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model