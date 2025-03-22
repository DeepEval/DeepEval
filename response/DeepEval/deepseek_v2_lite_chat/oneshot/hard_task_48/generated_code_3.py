import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def separable_conv(input_tensor, filters, kernel_size, strides):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters, 1, strides=strides, padding='same', activation='sigmoid')(conv)
        return conv

    def block(input_tensor):
        split = Lambda(lambda tensors: tf.split(tensors, [16, 16, 16], axis=1))(input_tensor)
        groups = [separable_conv(t, 64, (1, 1), (1, 1)) for t in split]
        output_tensor = Concatenate()(groups)
        return output_tensor

    input_layer = Input(shape=(32, 32, 3))
    block1_output = block(input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block1_output)

    def branch(input_tensor):
        conv = separable_conv(input_tensor, 64, (1, 1), (1, 1))
        avg_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(input_tensor)
        conv2 = separable_conv(avg_pool, 64, (3, 3), (1, 1))
        conv3 = separable_conv(input_tensor, 64, (5, 5), (1, 1))
        conv4 = separable_conv(avg_pool, 64, (1, 1), (1, 1))
        conv5 = separable_conv(input_tensor, 64, (1, 1), (1, 1))

        path1 = Concatenate()([conv, conv2, conv3, conv4])
        path2 = Concatenate()([conv2, conv3, conv4, conv5])
        path3 = Concatenate()([conv, conv2, conv4, conv5])
        path4 = Concatenate()([conv2, conv3, conv4, conv5])

        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block2_output = branch(block1_output)
    block2_output = BatchNormalization()(block2_output)
    block2_output = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(block2_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model