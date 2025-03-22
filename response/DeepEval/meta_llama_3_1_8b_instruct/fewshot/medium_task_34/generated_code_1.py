import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Conv2DTranspose

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: feature extraction through 3 pairs of <convolutional layer, max-pooling layer>
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_3)

    # Second part: enhancement of generalization capabilities through <convolutional layer, Dropout layer, convolutional layer>
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    dropout = Dropout(0.2)(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    # Third part: upsampling through 3 pairs of <convolutional layer, transposed convolutional layer> with skip connections
    up1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv2_2)
    skip1 = Concatenate()([up1, conv1_3])
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip1)
    up2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv3_1)
    skip2 = Concatenate()([up2, conv1_2])
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip2)
    up3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv3_2)
    skip3 = Concatenate()([up3, conv1_1])
    conv3_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip3)

    # Output layer: 1x1 convolutional layer for 10-class classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv3_3)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model