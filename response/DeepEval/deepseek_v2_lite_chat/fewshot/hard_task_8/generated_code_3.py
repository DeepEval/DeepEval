import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        conv2_1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_2 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        concat = Concatenate(axis=-1)([conv1_3, conv2_2])
        return concat

    def block_2(input_tensor):
        shape = Lambda(lambda tensor: tf.shape(tensor))(input_tensor)
        batch_norm = BatchNormalization()(input_tensor)
        reshape = Lambda(lambda tensor: tf.transpose(tensor, perm=[0, 2, 1, 3]))(batch_norm)
        reshape = Lambda(lambda tensor: tf.reshape(tensor, (-1, shape[1] // 2, shape[2] // 2, 4)))(reshape)
        swap = Lambda(lambda tensor: tf.transpose(tensor, perm=[0, 3, 1, 2]))(reshape)
        reshape2 = Lambda(lambda tensor: tf.reshape(tensor, shape))([swap])
        dense = Dense(units=128, activation='relu')(reshape2)
        output_layer = Dense(units=10, activation='softmax')(dense)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])