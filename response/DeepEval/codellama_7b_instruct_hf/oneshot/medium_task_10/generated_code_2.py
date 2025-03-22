import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First level
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    add = Concatenate()([maxpool1, maxpool2])
    batch_norm = BatchNormalization()(add)
    relu = LeakyReLU(alpha=0.2)(batch_norm)

    # Second level
    conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu)
    conv4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    maxpool4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)
    add2 = Concatenate()([maxpool3, maxpool4])
    batch_norm2 = BatchNormalization()(add2)
    relu2 = LeakyReLU(alpha=0.2)(batch_norm2)

    # Third level
    conv5 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu2)
    maxpool5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)
    batch_norm3 = BatchNormalization()(maxpool5)
    relu3 = LeakyReLU(alpha=0.2)(batch_norm3)

    # Final layer
    flatten = Flatten()(relu3)
    dense = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model