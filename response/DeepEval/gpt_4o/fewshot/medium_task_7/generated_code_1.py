import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Additional parallel convolutional layer directly processing the input
    direct_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Adding outputs of the convolutional paths
    added_outputs = Add()([conv1, conv2, conv3, direct_conv])

    # Fully connected layers
    flatten_layer = Flatten()(added_outputs)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model