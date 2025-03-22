import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    conv1_1 = block(input_layer, filters=32, kernel_size=(3, 3))
    conv1_2 = block(conv1_1, filters=32, kernel_size=(3, 3))
    conv1_3 = block(conv1_2, filters=32, kernel_size=(3, 3))

    # Second block
    conv2_1 = block(conv1_3, filters=64, kernel_size=(3, 3))
    conv2_2 = block(conv2_1, filters=64, kernel_size=(3, 3))
    conv2_3 = block(conv2_2, filters=64, kernel_size=(3, 3))

    # Third block
    conv3_1 = block(conv2_3, filters=128, kernel_size=(3, 3))
    conv3_2 = block(conv3_1, filters=128, kernel_size=(3, 3))
    conv3_3 = block(conv3_2, filters=128, kernel_size=(3, 3))

    # Direct convolutional processing
    direct_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    direct_conv = BatchNormalization()(direct_conv)
    direct_conv = ReLU()(direct_conv)

    # Adding outputs from all paths
    added_output = Add()([conv1_3, conv2_3, conv3_3, direct_conv])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model