import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.models import Model

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # First feature extraction block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    activation1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(activation1_1)
    activation1_2 = Activation('relu')(conv1_2)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2))(activation1_2)

    # Second feature extraction block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(max_pooling1)
    activation2_1 = Activation('relu')(conv2_1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(activation2_1)
    activation2_2 = Activation('relu')(conv2_2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2))(activation2_2)

    # Third feature extraction block
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(max_pooling2)
    activation3_1 = Activation('relu')(conv3_1)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(activation3_1)
    activation3_2 = Activation('relu')(conv3_2)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(activation3_2)
    activation3_3 = Activation('relu')(conv3_3)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2))(activation3_3)

    # Fourth feature extraction block
    conv4_1 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(max_pooling3)
    activation4_1 = Activation('relu')(conv4_1)
    conv4_2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(activation4_1)
    activation4_2 = Activation('relu')(conv4_2)
    conv4_3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(activation4_2)
    activation4_3 = Activation('relu')(conv4_3)
    max_pooling4 = MaxPooling2D(pool_size=(2, 2))(activation4_3)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(max_pooling4)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model