import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(batch_norm1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(batch_norm2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu2)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(batch_norm3)

    # Parallel branch
    parallel_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    parallel_batch_norm1 = BatchNormalization()(parallel_conv1)
    parallel_relu1 = Activation('relu')(parallel_batch_norm1)

    parallel_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(parallel_relu1)
    parallel_batch_norm2 = BatchNormalization()(parallel_conv2)
    parallel_relu2 = Activation('relu')(parallel_batch_norm2)

    parallel_conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(parallel_relu2)
    parallel_batch_norm3 = BatchNormalization()(parallel_conv3)
    parallel_relu3 = Activation('relu')(parallel_batch_norm3)

    # Concatenate outputs from all paths
    output_tensor = Concatenate()([relu3, parallel_relu3])

    # Add spatial dimensions
    output_tensor = keras.layers.GlobalAveragePooling2D()(output_tensor)

    # Classification
    dense1 = Dense(units=256, activation='relu')(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model