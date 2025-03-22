import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Initial 1x1 convolution layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)

    # Branch 2: Max pooling, followed by 3x3 convolution and upsampling
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3: Max pooling, followed by 3x3 convolution and upsampling
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv)

    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Final 1x1 convolution layer
    final_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(final_conv)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model