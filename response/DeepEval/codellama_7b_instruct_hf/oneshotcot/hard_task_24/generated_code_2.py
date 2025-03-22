import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 initial convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 1: extract local features through 3x3 convolutional layer
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch 2: downsample through max pooling layer
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Branch 3: restore to original size through upsampling layer
    upsampling2 = UpSampling2D(size=(2, 2))(max_pooling2)
    conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(upsampling2)

    # Concatenate outputs of all branches
    concatenate_layer = Concatenate()([conv2, max_pooling2, upsampling2, conv3])

    # 1x1 convolutional layer to fuse features
    conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(concatenate_layer)

    # Batch normalization and flatten layer
    batch_norm = BatchNormalization()(conv4)
    flatten_layer = Flatten()(batch_norm)

    # Three fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model