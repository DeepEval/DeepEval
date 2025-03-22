import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def conv_block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = keras.activations.relu(batch_norm)
        return relu

    # Creating three separate blocks
    block1_output = conv_block(input_layer, filters=32)
    block2_output = conv_block(block1_output, filters=64)
    block3_output = conv_block(block2_output, filters=128)

    # Direct parallel branch
    parallel_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_batch_norm = BatchNormalization()(parallel_conv)
    parallel_relu = keras.activations.relu(parallel_batch_norm)

    # Adding all paths together
    aggregated_output = Add()([block3_output, parallel_relu])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(aggregated_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=dense2)

    return model