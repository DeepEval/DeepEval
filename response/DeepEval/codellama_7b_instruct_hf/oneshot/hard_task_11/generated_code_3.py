import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 convolution pathway
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 1x3 and 3x1 convolutions pathway
    conv2 = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_concatenate = Concatenate()([conv1, conv2, conv3])

    # 1x1 convolution pathway
    conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(parallel_concatenate)

    # Global Average Pooling
    gavg_pool = GlobalAveragePooling2D()(conv4)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(gavg_pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model