from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)

    # Branch 1: Local feature extraction
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)

    # Branch 2: Downsampling and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Downsampling and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(conv1)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate branches
    concat_layer = concatenate([branch1, branch2, branch3])

    # Fusion layer
    fusion = Conv2D(32, (1, 1), padding='same', activation='relu')(concat_layer)

    # Fully connected layers
    flatten = Flatten()(fusion)
    dense1 = Dense(256, activation='relu')(flatten)
    dense2 = Dense(128, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    # Model creation
    model = Model(inputs=input_img, outputs=output)

    return model