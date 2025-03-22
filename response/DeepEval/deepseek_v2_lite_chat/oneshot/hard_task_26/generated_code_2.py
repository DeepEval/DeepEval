import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Branch path
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(pool1)
    branch_conv1 = BatchNormalization()(branch_conv1)
    branch_conv1 = Activation('relu')(branch_conv1)

    branch_upsample1 = UpSampling2D(size=(2, 2))(branch_conv1)
    branch_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch_upsample1)
    branch_conv2 = BatchNormalization()(branch_conv2)
    branch_conv2 = Activation('relu')(branch_conv2)

    branch_upsample2 = UpSampling2D(size=(2, 2))(branch_conv2)
    branch_conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch_upsample2)
    branch_conv3 = BatchNormalization()(branch_conv3)
    branch_conv3 = Activation('relu')(branch_conv3)

    # Concatenate outputs from all paths
    concat = Concatenate()([conv2, branch_conv2, branch_conv3])

    # Final 1x1 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(conv4)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()