import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # 3x3 convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # 5x5 convolutions
    conv3_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv3_1)

    # 3x3 max pooling
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(pool1)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(pool2)

    # concatenate feature maps
    concatenated = Concatenate()([conv1_2, conv2_2, conv3_2, pool3])

    # flatten and dense layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # compile the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model