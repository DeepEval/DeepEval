import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Two 1x1 convolutional layers
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_dropout = Dropout(rate=0.2)(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_dropout)
    conv2_dropout = Dropout(rate=0.2)(conv2)

    # 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2_dropout)
    conv3_dropout = Dropout(rate=0.2)(conv3)

    # 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3_dropout)
    conv4_dropout = Dropout(rate=0.2)(conv4)

    # Restore number of channels to match input's channel count
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4_dropout)
    conv5_dropout = Dropout(rate=0.2)(conv5)

    # Add processed features to original input
    concatenate = Concatenate()([input_layer, conv5_dropout])

    # Flatten layer
    flatten = Flatten()(concatenate)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model