import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolution branch
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Three separate branches
    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch 2: 1x1 conv -> Max Pooling -> 3x3 conv
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(1, 1))(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 5x5 convolutional layer
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Upsampling to restore size
    upsample2 = UpSampling2D(size=(2, 2))(branch2)
    concat1 = Concatenate()([branch3, upsample2])
    upsample1 = UpSampling2D(size=(2, 2))(concat1)

    # Final 1x1 convolution for fusing branches
    conv_fuse = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(upsample1)

    # Pass through another 1x1 convolutional layer
    batch_norm_fuse = BatchNormalization()(conv_fuse)
    flatten_layer = Flatten()(batch_norm_fuse)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model