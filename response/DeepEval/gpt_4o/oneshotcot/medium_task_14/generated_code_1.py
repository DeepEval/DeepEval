import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(bn2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(bn3)

    # Parallel branch processing input directly
    parallel_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_bn = BatchNormalization()(parallel_conv)
    parallel_relu = Activation('relu')(parallel_bn)

    # Add outputs from all paths
    added_outputs = Add()([relu1, relu2, relu3, parallel_relu])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(added_outputs)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model