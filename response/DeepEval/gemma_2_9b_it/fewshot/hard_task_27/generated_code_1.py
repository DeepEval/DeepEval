import keras
from keras.layers import Input, Conv2D, Lambda, Dense, Add, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 7x7 depthwise separable convolution with layer normalization
    depthwise_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same',
                           use_bias=False, kernel_initializer='he_normal')(input_layer)
    batch_norm = BatchNormalization()(depthwise_conv)

    # Two fully connected layers for channel-wise feature transformation
    dense1 = Dense(units=32, activation='relu')(batch_norm)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Combine original input with processed features
    adding_layer = Add()([input_layer, dense2])

    # Final two fully connected layers for classification
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model