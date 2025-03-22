import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Dropout, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Encoder
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    drop3 = Dropout(0.5)(conv3)

    # Decoder
    deconv1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(drop3)
    deconv1 = Add()([deconv1, conv2])
    deconv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(deconv1)
    deconv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(deconv1)

    deconv2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(deconv1)
    deconv2 = Add()([deconv2, conv1])
    deconv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(deconv2)
    deconv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(deconv2)

    # Output layer
    output_layer = Conv2D(10, (1, 1), activation='softmax')(deconv2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()