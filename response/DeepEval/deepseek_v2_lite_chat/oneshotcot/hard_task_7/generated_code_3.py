import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)

    # Block 1
    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(x, [32, 32, 32, 32], axis=-1))(input_tensor)
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        depthwise = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')(split[1])
        pointwise = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
        group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[2])
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(split[3])
        output_tensor = Concatenate(axis=-1)([group1, group2, pointwise, maxpool])

        return output_tensor
    
    block1_output = block1(conv1)
    batch_norm1 = BatchNormalization()(block1_output)
    flatten1 = Flatten()(batch_norm1)

    dense1 = Dense(units=128, activation='relu')(flatten1)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Block 2
    def block2(input_tensor):
        input_shape = input_tensor.shape
        reshape_tensor = Lambda(lambda x: tf.reshape(x, (input_shape[0]*input_shape[1], input_shape[2]*input_shape[3], input_shape[4])))(input_tensor)
        reshape_tensor = Lambda(lambda x: tf.transpose(x, (2, 1, 3, 4)))(reshape_tensor)
        reshape_tensor = Lambda(lambda x: tf.reshape(x, (input_shape[0], input_shape[1], input_shape[2]*input_shape[3], input_shape[4])))(reshape_tensor)
        dense_output = Dense(units=128, activation='relu')(reshape_tensor)

        return dense_output
    
    block2_output = block2(block1_output)
    output = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model