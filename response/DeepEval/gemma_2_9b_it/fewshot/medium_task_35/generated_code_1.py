import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)

    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Stage 2: Feature Extraction
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    dropout1 = Dropout(0.25)(conv2_1)

    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2_2)

    # Stage 3: Upsampling and Reconstruction
    up1 = UpSampling2D(size=(2, 2))(dropout2)
    concat1 = Concatenate()([up1, conv1_2]) 
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)

    up2 = UpSampling2D(size=(2, 2))(conv3_1)
    concat2 = Concatenate()([up2, conv1_1])
    conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)

    # Final Classification Layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv3_2)


    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model