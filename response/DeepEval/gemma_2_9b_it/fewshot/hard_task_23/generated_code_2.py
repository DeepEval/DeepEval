import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 Convolutional
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch 1: Local Feature Extraction
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2_1)

    # Branch 2: Downsampling & Upsampling
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    upsample1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(conv3_1)

    # Branch 3: Downsampling & Upsampling
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv4_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool2)
    upsample2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(conv4_1)

    # Concatenate outputs
    merged = Concatenate()([conv2_2, upsample1, upsample2])

    # Final 1x1 Convolution
    merged = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(merged)

    # Flatten and classify
    flatten = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model