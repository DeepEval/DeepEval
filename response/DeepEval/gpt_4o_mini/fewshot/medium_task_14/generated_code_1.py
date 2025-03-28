import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch_norm2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch_norm3)

    # Parallel branch
    parallel_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    parallel_batch_norm = BatchNormalization()(parallel_conv)
    parallel_relu = ReLU()(parallel_batch_norm)

    # Adding outputs from all paths
    combined_output = Add()([relu1, relu2, relu3, parallel_relu])

    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model