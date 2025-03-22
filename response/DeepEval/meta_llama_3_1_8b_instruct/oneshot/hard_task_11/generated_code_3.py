import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense
from keras.layers import Activation, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    conv_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_path = BatchNormalization()(conv_path)

    # Parallel branch
    parallel_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_path = BatchNormalization()(parallel_path)
    parallel_path = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(parallel_path)
    parallel_path = BatchNormalization()(parallel_path)
    parallel_path = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(parallel_path)
    parallel_path = BatchNormalization()(parallel_path)

    # Concatenate and apply 1x1 convolution
    output_path = Concatenate()([conv_path, parallel_path])
    output_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(output_path)
    output_path = BatchNormalization()(output_path)
    output_path = Activation('relu')(output_path)

    # Direct connection
    direct_connection = input_layer

    # Fusion
    fusion = Add()([output_path, direct_connection])

    # Average pooling
    avg_pool = AveragePooling2D(pool_size=(8, 8))(fusion)

    # Flatten and dense layers
    flatten_layer = Flatten()(avg_pool)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model