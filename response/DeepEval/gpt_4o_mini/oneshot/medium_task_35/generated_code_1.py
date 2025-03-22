import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # First stage of convolutions and max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second stage of convolutions and max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional convolution and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.5)(conv3)

    # Upsampling stage with skip connections
    up1 = UpSampling2D(size=(2, 2))(conv3)
    skip1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv2)
    merge1 = Concatenate()([up1, skip1])
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(merge1)
    conv4 = BatchNormalization()(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(conv1)
    merge2 = Concatenate()([up2, skip2])
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(merge2)
    conv5 = BatchNormalization()(conv5)

    # Final 1x1 convolution for classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5)
    flatten_layer = Flatten()(output_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=flatten_layer)

    return model