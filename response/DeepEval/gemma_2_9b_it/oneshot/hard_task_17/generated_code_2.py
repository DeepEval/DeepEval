import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = keras.layers.Reshape((32, 32, 64))(x)  
    weighted_features = x * input_layer

    # Block 2
    y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(y)

    # Fusion
    output = Add()([weighted_features, y])

    output = Flatten()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model