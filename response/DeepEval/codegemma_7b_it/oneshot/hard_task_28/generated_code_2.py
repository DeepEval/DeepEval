import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn1)
    bn2 = BatchNormalization()(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn2)

    # Branch Path
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Addition Operation
    output = Add()([conv3, branch])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model