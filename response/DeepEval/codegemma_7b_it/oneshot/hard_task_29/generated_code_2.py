import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First block: Main path and branch path
    conv_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv_main = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv_main)

    conv_branch = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    concat_main_branch = keras.layers.Add()([conv_main, conv_branch])

    # Second block: Max pooling layers
    max_pool_1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(concat_main_branch)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(concat_main_branch)
    max_pool_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(concat_main_branch)

    # Flatten and concatenate outputs of second block
    flatten_max_pool_1 = Flatten()(max_pool_1)
    flatten_max_pool_2 = Flatten()(max_pool_2)
    flatten_max_pool_3 = Flatten()(max_pool_3)
    concat_max_pool = keras.layers.Concatenate()([flatten_max_pool_1, flatten_max_pool_2, flatten_max_pool_3])

    # Fully connected layers for classification
    dense_layer_1 = Dense(units=64, activation='relu')(concat_max_pool)
    dense_layer_2 = Dense(units=32, activation='relu')(dense_layer_1)
    output_layer = Dense(units=10, activation='softmax')(dense_layer_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model