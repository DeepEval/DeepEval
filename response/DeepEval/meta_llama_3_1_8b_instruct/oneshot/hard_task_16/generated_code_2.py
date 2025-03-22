import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras import regularizers

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(x, 3, -1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
        concat = Concatenate()([conv1, conv2, conv3])
        return concat

    block1_output = block1(input_layer)
    transition_conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    def block2(input_tensor):
        global_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        weight = Dense(96, activation='relu')(global_max_pool)
        weight = Dense(96, activation='relu')(weight)
        weight = tf.reshape(weight, (-1, 1, 1, 96))
        return input_tensor * weight + input_tensor

    block2_output = block2(transition_conv)
    branch_output = block1_output
    main_path_output = block2_output
    added = layers.Add()([main_path_output, branch_output])
    output_layer = Dense(10, activation='softmax')(added)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()