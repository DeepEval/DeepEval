import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Define branches
    # Branch 1: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    # Branch 2: MaxPooling -> 3x3 convolution -> UpSampling
    branch2_pool = MaxPooling2D(pool_size=(2, 2))(initial_conv)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3: MaxPooling -> 3x3 convolution -> UpSampling
    branch3_pool = MaxPooling2D(pool_size=(2, 2))(initial_conv)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv)

    # Step 4: Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Step 5: Add another 1x1 convolutional layer
    conv_after_concat = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Step 6: Flatten the result
    flatten_layer = Flatten()(conv_after_concat)

    # Step 7: Add dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Step 8: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model