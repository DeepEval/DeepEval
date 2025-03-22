import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Stage 2: Downsampling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional Convolution and Dropout
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    dropout = Dropout(0.5)(conv3)

    # Stage 1: Upsampling
    up1 = UpSampling2D(size=(2, 2))(dropout)
    upconv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(up1)
    # Skip connection
    merge1 = Concatenate()([upconv1, conv2])

    # Stage 2: Upsampling
    up2 = UpSampling2D(size=(2, 2))(merge1)
    upconv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up2)
    # Skip connection
    merge2 = Concatenate()([upconv2, conv1])

    # Output Layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(merge2)

    # Construct the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model