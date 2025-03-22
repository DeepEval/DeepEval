from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(initial_conv)

    # Branch 2: MaxPooling, 3x3 Convolution, and UpSampling
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch2_pool)
    branch2_up = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3: MaxPooling, 3x3 Convolution, and UpSampling
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch3_pool)
    branch3_up = UpSampling2D(size=(2, 2))(branch3_conv)

    # Concatenate outputs from the three branches
    concatenated = Concatenate()([branch1, branch2_up, branch3_up])

    # Additional 1x1 convolutional layer after concatenation
    conv_after_concat = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(concatenated)

    # Flattening and fully connected layers
    flatten_layer = Flatten()(conv_after_concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model