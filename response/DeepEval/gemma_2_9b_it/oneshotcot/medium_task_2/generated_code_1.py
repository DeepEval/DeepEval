import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3)) 

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    # Branch Path
    branch_conv = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)

    # Concatenate paths
    combined_features = Concatenate()([max_pool, branch_conv])

    # Flatten and Dense Layers
    flatten = Flatten()(combined_features)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model