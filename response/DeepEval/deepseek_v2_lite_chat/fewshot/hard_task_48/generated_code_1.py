import keras
from keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPool2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split[2])
        concat = Concatenate(axis=-1)([conv1, conv2, conv3])
        return concat

    def block_2(input_tensor):
        conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        avg_pool = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(avg_pool)
        avg_pool2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(avg_pool2)
        avg_pool3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        conv4 = SeparableConv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same')(avg_pool3)
        concat = Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
        return concat

    block1_output = block_1(input_tensor=input_layer)
    dense1 = BatchNormalization()(block1_output)
    dense2 = Flatten()(dense1)

    block2_output = block_2(input_tensor=dense2)
    dense3 = BatchNormalization()(block2_output)
    dense4 = Flatten()(dense3)

    output_layer = Dense(units=10, activation='softmax')(dense4)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model