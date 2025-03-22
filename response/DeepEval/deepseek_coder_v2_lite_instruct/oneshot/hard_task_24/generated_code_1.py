import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial 1x1 convolutional layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch 1: Local feature extraction with 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)

    # Branch 2: Sequential operations including max pooling, 3x3 convolutional, and upsampling
    branch2_pool = MaxPooling2D(pool_size=(2, 2))(conv1x1)
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv)  # Upsampling to restore size

    # Branch 3: Similar to Branch 2 but without initial max pooling
    branch3_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv)  # Upsampling to restore size

    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Further processing through another 1x1 convolutional layer
    conv1x1_after_concat = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)

    # Flatten the output and pass through fully connected layers
    flatten_layer = Flatten()(conv1x1_after_concat)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model