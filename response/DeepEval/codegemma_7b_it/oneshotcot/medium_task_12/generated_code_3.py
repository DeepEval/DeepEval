import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)

    # Block 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch_norm2)

    # Block 3
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch_norm3)

    # Concatenate outputs of blocks
    concat = Concatenate()([relu1, relu2, relu3])

    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model