import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dense, Flatten, tf

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        group1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[0])
        group1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(group1)
        group1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group1)

        group2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[1])
        group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(group2)
        group2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group2)

        group3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[2])
        group3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(group3)
        group3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group3)

        return Concatenate()([group1, group2, group3])

    block1_output = block1(input_tensor)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(block1_output)

    # Block 2
    def block2(input_tensor):
        global_max_pool = MaxPooling2D(pool_size=(8, 8), strides=(8, 8))(transition_conv)
        
        dense1 = Dense(units=64)(global_max_pool)
        dense2 = Dense(units=transition_conv.shape[-1])(dense1)
        weights = tf.reshape(dense2, (-1, 8, 8, 64))

        return tf.multiply(transition_conv, weights)

    block2_output = block2(transition_conv)

    # Branch
    branch = input_tensor

    # Main Path + Branch
    output = tf.add(block2_output, branch)

    # Final Classification Layer
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)

    return model