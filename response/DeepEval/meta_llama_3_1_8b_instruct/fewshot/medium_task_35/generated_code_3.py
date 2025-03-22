import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Concatenate, Dropout, Conv2DTranspose
from keras.layers import Activation, BatchNormalization

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional block 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    # Convolutional block 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Feature extraction block
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)

    # Upsampling block 1
    up1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    up1 = BatchNormalization()(up1)
    skip_connection1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    skip_connection1 = BatchNormalization()(skip_connection1)
    merged1 = Add()([up1, skip_connection1])
    merged1 = Activation('relu')(merged1)

    # Upsampling block 2
    up2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(merged1)
    up2 = BatchNormalization()(up2)
    skip_connection2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    skip_connection2 = BatchNormalization()(skip_connection2)
    merged2 = Add()([up2, skip_connection2])
    merged2 = Activation('relu')(merged2)

    # Classification block
    conv4 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(merged2)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    drop1 = Dropout(0.5)(conv4)
    conv5 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(drop1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    drop2 = Dropout(0.5)(conv5)
    conv6 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(drop2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    # Output layer
    output_layer = Conv2DTranspose(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv6)
    output_layer = Activation('softmax')(output_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model