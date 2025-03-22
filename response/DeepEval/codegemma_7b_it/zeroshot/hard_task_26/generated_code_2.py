from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense

def dl_model():

    # Input layer
    img = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (1, 1), padding='same')(img)
    branch1 = Conv2D(32, (3, 3), padding='same')(x)
    branch2 = MaxPooling2D((2, 2), padding='same')(x)
    branch2 = Conv2D(32, (3, 3), padding='same')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)
    branch3 = MaxPooling2D((2, 2), padding='same')(x)
    branch3 = Conv2D(32, (3, 3), padding='same')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)
    branch_concat = concatenate([branch1, branch2, branch3])
    x = Conv2D(32, (1, 1), padding='same')(branch_concat)

    # Branch path
    y = Conv2D(32, (1, 1), padding='same')(img)

    # Concatenate main and branch outputs
    concat = concatenate([x, y])

    # Fully connected layers
    flatten = Flatten()(concat)
    dense = Dense(10, activation='softmax')(flatten)

    # Create model
    model = Model(inputs=img, outputs=dense)

    return model