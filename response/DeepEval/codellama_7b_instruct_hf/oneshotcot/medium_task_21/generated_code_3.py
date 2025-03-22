import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # 1x1 convolution branch
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.2)(conv1)

    # 1x1 convolution followed by 3x3 convolution branch
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    dropout2 = Dropout(rate=0.2)(conv3)

    # 1x1 convolution followed by two consecutive 3x3 convolutions branch
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    dropout3 = Dropout(rate=0.2)(conv6)

    # Average pooling followed by 1x1 convolution branch
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    conv7 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    dropout4 = Dropout(rate=0.2)(conv7)

    # Concatenate outputs from all branches
    outputs = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6, conv7])
    outputs = Dropout(rate=0.2)(outputs)

    # Flatten and pass through three fully connected layers for classification
    flatten = Flatten()(outputs)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model