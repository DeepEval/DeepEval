import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial 1x1 convolution
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)

    # Branch 2: Max pooling followed by 3x3 convolution and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max pooling followed by 3x3 convolution and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Final 1x1 convolution
    final_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Flatten and fully connected layers
    flatten = Flatten()(final_conv)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model