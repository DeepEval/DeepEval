import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act1)

    # Second block
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act2)

    # Third block
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act3)

    # Parallel branch
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act4)

    # Aggregate outputs
    add = Add()([pool3, pool4])

    # Flatten and pass through fully connected layers
    flatten = Flatten()(add)
    fc1 = Dense(units=128, activation='relu')(flatten)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=fc2)

    return model