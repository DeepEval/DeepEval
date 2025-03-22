import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = Dropout(0.2)(path1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Dropout(0.2)(path2)
    path2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Dropout(0.2)(path2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Dropout(0.2)(path3)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Dropout(0.2)(path3)
    path3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Dropout(0.2)(path3)

    # Branch 4: Average pooling followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Dropout(0.2)(path4)

    # Concatenate outputs from all branches
    concat = Concatenate()([path1, path2, path3, path4])

    # Batch normalization and flatten
    concat = BatchNormalization()(concat)
    flatten_layer = Flatten()(concat)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=128, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model