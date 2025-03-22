import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First block
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv1_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    output_tensor = Concatenate()([conv1_1, conv1_2, conv1_3, conv1_4])
    batch_norm1 = BatchNormalization()(output_tensor)

    # Second block
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv2_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv2_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    output_tensor = Concatenate()([conv2_1, conv2_2, conv2_3, conv2_4])
    batch_norm2 = BatchNormalization()(output_tensor)

    # Third block
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
    conv3_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
    conv3_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
    output_tensor = Concatenate()([conv3_1, conv3_2, conv3_3, conv3_4])
    batch_norm3 = BatchNormalization()(output_tensor)

    # Flatten and fully connected layers
    flatten = Flatten()(batch_norm3)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model