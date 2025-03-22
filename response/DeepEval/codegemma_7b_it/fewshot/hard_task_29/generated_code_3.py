import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Branch Path
    conv2_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Combine Outputs
    output_layer = Add()([conv1_2, conv2_1])

    # Second Block
    max_pool_1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(output_layer)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output_layer)
    max_pool_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(output_layer)

    # Flatten and Concatenate
    flatten_1 = Flatten()(max_pool_1)
    flatten_2 = Flatten()(max_pool_2)
    flatten_3 = Flatten()(max_pool_3)
    concat_output = Concatenate()([flatten_1, flatten_2, flatten_3])

    # Fully Connected Layers
    dense_1 = Dense(units=128, activation='relu')(concat_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model