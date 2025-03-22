import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Part 1: Feature extraction with convolutional and max-pooling layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Part 2: Processing with convolutional layers and Dropout
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    drop1 = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(drop1)

    # Part 3: Upsampling with convolutional and transposed convolutional layers
    upsample1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    skip1 = Concatenate()([upsample1, conv3])
    
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    
    upsample2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    skip2 = Concatenate()([upsample2, conv2])
    
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip2)
    
    upsample3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    skip3 = Concatenate()([upsample3, conv1])
    
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip3)

    # Final output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv8)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model