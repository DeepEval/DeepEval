import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(conv_layer)
    flat1 = Flatten()(pool1)

    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_layer)
    flat2 = Flatten()(pool2)

    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(conv_layer)
    flat3 = Flatten()(pool3)

    combined_features = Concatenate()([flat1, flat2, flat3])

    dense1 = Dense(units=128, activation='relu')(combined_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model