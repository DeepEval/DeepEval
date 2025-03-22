import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    drop1_1 = Dropout(0.2)(conv1_1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    drop2_2 = Dropout(0.2)(conv2_2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_3)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_3)
    drop3_3 = Dropout(0.2)(conv3_3)

    # Branch 4: Average pooling followed by 1x1 convolution
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    conv1_4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    drop4_4 = Dropout(0.2)(conv1_4)

    # Concatenate outputs from all branches
    concat_layer = concatenate([drop1_1, drop2_2, drop3_3, drop4_4])

    # Fully connected layers for classification
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model