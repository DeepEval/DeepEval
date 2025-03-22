import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, AveragePooling2D, Reshape
from keras.layers import Add, Activation, LeakyReLU, PReLU
from keras import regularizers

def dl_model():    

    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local feature extraction
    branch1 = conv
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Branch 2: Downsampling followed by upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: Similar to branch 2 but with a different initial convolutional layer
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate and refine using a 1x1 convolutional layer
    concat = Concatenate()([branch1, branch2, branch3])
    refined = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Flatten the output
    flat = Flatten()(refined)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flat)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model