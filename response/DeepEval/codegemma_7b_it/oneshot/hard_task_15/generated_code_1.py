import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, concatenate, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    gap = GlobalAveragePooling2D()(conv1)
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=input_layer.shape[3], activation='sigmoid')(fc1)
    fc2 = keras.layers.Reshape((input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]))(fc2)
    concat_main = keras.layers.Multiply()([input_layer, fc2])

    # Branch path
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    concat_branch = concatenate([input_layer, conv2])

    # Combine main and branch paths
    concat = Add()([concat_main, concat_branch])

    # Fully connected layers
    flatten = keras.layers.Flatten()(concat)
    fc3 = Dense(units=128, activation='relu')(flatten)
    fc4 = Dense(units=10, activation='softmax')(fc3)

    model = keras.Model(inputs=input_layer, outputs=fc4)

    return model