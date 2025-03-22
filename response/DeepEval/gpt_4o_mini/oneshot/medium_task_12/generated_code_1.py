import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch_norm2)
    block2_output = Concatenate()([relu1, relu2])  # Concatenating output from Block 1 and Block 2

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(block2_output)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch_norm3)
    block3_output = Concatenate()([block2_output, relu3])  # Concatenating output from Block 2 and Block 3

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model