from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(64, (1, 1), padding='same')(input_layer)
    branch1 = Conv2D(64, (3, 3), padding='same')(conv1)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    merged = Concatenate()([branch1, branch2, branch3])
    conv2 = Conv2D(16, (1, 1), padding='same')(merged)
    flatten = Flatten()(conv2)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(10, activation='softmax')(dense1)
    model = Model(inputs=input_layer, outputs=dense2)

    return model