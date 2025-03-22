import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First stage of convolutions and max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Second stage of convolutions and max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Additional convolutional layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    dropout = Dropout(0.5)(conv3)

    # Upsampling with skip connections
    upsample1 = UpSampling2D(size=(2, 2))(dropout)
    skip1 = Concatenate()([upsample1, conv2])
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv4)

    upsample2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Concatenate()([upsample2, conv1])
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip2)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv5)

    # Final output layer with 1x1 convolution
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5)

    # Flatten to prepare for classification
    flatten_layer = Flatten()(output_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=flatten_layer)

    return model