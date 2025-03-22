import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 convolution branch
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    flatten1 = Flatten()(dropout1)

    # 1x1 convolution followed by 3x3 convolution branch
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    dropout2 = Dropout(0.2)(conv3)
    flatten2 = Flatten()(dropout2)

    # 1x1 convolution followed by two consecutive 3x3 convolutions branch
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    dropout3 = Dropout(0.2)(conv6)
    flatten3 = Flatten()(dropout3)

    # Average pooling followed by 1x1 convolution branch
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv7 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
    dropout4 = Dropout(0.2)(conv7)
    flatten4 = Flatten()(dropout4)

    # Concatenate outputs from all branches
    concatenated = keras.layers.concatenate([flatten1, flatten2, flatten3, flatten4])

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model