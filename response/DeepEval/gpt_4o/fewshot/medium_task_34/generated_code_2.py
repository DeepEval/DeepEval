import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature extraction with 3 pairs of convolution and max-pooling layers
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    # Part 2: Processing with convolution, dropout, and another convolution
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling with skip connections
    up_conv1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
    skip1 = Add()([up_conv1, conv3])

    up_conv2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(skip1)
    skip2 = Add()([up_conv2, conv2])

    up_conv3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(skip2)
    skip3 = Add()([up_conv3, conv1])

    # Final 1x1 convolution to generate probability output for 10 classes
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(skip3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model