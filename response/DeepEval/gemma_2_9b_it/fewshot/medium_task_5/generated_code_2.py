import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1_1)
    pool1_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)
    
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1_2)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2_1)
    pool2_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Branch Path
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3_1)
    pool3_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_2)

    # Combine paths
    combined_output = Add()([pool2_2, pool3_2])

    flatten_layer = Flatten()(combined_output)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model