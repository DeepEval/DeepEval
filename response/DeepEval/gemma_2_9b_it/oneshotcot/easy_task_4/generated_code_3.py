import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 2
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    conv7 = Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv7)

    # Flatten and dense layers
    flatten = Flatten()(pool2)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model