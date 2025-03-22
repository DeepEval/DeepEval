import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: local features
    conv1_1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv1_2 = Conv2D(32, (3, 3), activation='relu')(conv1_1)
    maxpool1 = MaxPooling2D((2, 2))(conv1_2)

    # Branch 2: downsampling and upsampling
    conv2_1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool2 = MaxPooling2D((2, 2))(conv2_1)
    upsample1 = UpSampling2D((2, 2))(maxpool2)
    conv2_2 = Conv2D(32, (3, 3), activation='relu')(upsample1)

    # Branch 3: downsampling and upsampling
    conv3_1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool3 = MaxPooling2D((2, 2))(conv3_1)
    upsample2 = UpSampling2D((2, 2))(maxpool3)
    conv3_2 = Conv2D(32, (3, 3), activation='relu')(upsample2)

    # Fuse branches
    concatenated = Concatenate()([maxpool1, conv2_2, conv3_2])

    # Global features
    conv4_1 = Conv2D(32, (3, 3), activation='relu')(concatenated)
    maxpool4 = MaxPooling2D((2, 2))(conv4_1)
    flattened = Flatten()(maxpool4)

    # Output layer
    output = Dense(10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model