import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: Main path and branch path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    adding_layer = Add()([main_path, branch_path])

    # Second block: Max pooling layers with varying scales
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(adding_layer)
    flatten1 = Flatten()(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(adding_layer)
    flatten2 = Flatten()(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(adding_layer)
    flatten3 = Flatten()(maxpool3)

    concat_layer = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model