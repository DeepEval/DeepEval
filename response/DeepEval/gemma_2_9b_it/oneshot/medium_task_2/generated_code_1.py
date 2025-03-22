import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    max_pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_main)

    # Branch Path
    conv1_branch = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate Features
    combined_features = Concatenate()([max_pool_main, conv1_branch])

    # Flatten and Dense Layers
    flatten_layer = Flatten()(combined_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model