import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1_1)

    # Branch path
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Combine paths
    adding_layer = Add()([conv1_2, branch_path])

    # Second block
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(adding_layer)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(adding_layer)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(adding_layer)

    # Flatten and concatenate
    flatten_layer1 = Flatten()(max_pooling1)
    flatten_layer2 = Flatten()(max_pooling2)
    flatten_layer3 = Flatten()(max_pooling3)
    concat_layer = Concatenate()([flatten_layer1, flatten_layer2, flatten_layer3])

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model