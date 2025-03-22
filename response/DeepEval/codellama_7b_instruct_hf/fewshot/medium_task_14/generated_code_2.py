import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)

    # Third block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)

    # Parallel branch
    parallel_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu3)

    # Aggregate outputs from all blocks
    aggregated_output = Add()([relu1, relu2, relu3, parallel_branch])

    # Flatten and pass through two fully connected layers
    flattened = Flatten()(aggregated_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model