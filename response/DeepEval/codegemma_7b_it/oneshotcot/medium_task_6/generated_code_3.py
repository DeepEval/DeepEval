import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolution
    conv_init = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Three parallel blocks
    block1_output = block(conv_init, filters=64)
    block2_output = block(conv_init, filters=128)
    block3_output = block(conv_init, filters=256)

    # Add outputs of parallel blocks to initial convolution's output
    added_output = concatenate([conv_init, block1_output, block2_output, block3_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Define block function
def block(input_tensor, filters):

    path = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path = BatchNormalization()(path)
    path = ReLU()(path)

    return path