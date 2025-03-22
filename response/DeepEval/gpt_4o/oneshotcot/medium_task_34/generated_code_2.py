import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First part: Feature extraction with 3 pairs of convolutional and max-pooling layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Second part: Enhance generalization with convolutional, dropout, and convolutional layers
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    # Third part: Upsampling with transposed convolutional layers and skip connections
    upconv1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    skip1 = Add()([upconv1, conv3])

    upconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(skip1)
    skip2 = Add()([upconv2, conv2])

    upconv3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(skip2)
    skip3 = Add()([upconv3, conv1])

    # Final 1x1 convolutional layer to produce the probability output for 10 classes
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(skip3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model