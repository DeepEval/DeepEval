import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        relu1 = ReLU()(batch_norm1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(relu1)

        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool1)
        batch_norm2 = BatchNormalization()(conv2)
        relu2 = ReLU()(batch_norm2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(relu2)

        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool2)
        batch_norm3 = BatchNormalization()(conv3)
        relu3 = ReLU()(batch_norm3)

        return Concatenate()([relu1, relu2, relu3])

    def block_2(input_tensor):
        conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm4 = BatchNormalization()(conv4)
        relu4 = ReLU()(batch_norm4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(relu4)

        conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool3)
        batch_norm5 = BatchNormalization()(conv5)
        relu5 = ReLU()(batch_norm5)
        pool4 = MaxPooling2D(pool_size=(2, 2))(relu5)

        conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool4)
        batch_norm6 = BatchNormalization()(conv6)
        relu6 = ReLU()(batch_norm6)

        return Concatenate()([relu4, relu5, relu6])

    def block_3(input_tensor):
        flat = Flatten()(input_tensor)
        dense1 = Dense(units=512, activation='relu')(flat)
        dense2 = Dense(units=10, activation='softmax')(dense1)

        return dense2

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)

    model = keras.Model(inputs=input_layer, outputs=block3_output)

    return model