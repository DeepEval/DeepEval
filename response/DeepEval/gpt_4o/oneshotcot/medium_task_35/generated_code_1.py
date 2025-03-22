from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First stage of convolution and pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second stage of convolution and pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    dropout = Dropout(0.5)(conv3)

    # Upsampling and concatenation with skip connections
    up1 = UpSampling2D(size=(2, 2))(dropout)
    up1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(up1)
    skip1 = Concatenate()([up1, conv2])

    up2 = UpSampling2D(size=(2, 2))(skip1)
    up2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up2)
    skip2 = Concatenate()([up2, conv1])

    # Output layer with 1x1 convolution
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(skip2)

    # Construct model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model