import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, Dense, Reshape, Multiply
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = block1(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(block1_output)

    # Block 2
    def block2(input_tensor):
        pool_output = MaxPooling2D(pool_size=(7, 7), strides=(7, 7), padding='valid')(input_tensor)
        dense1 = Dense(units=pool_output.shape[-1], activation='relu')(tf.reshape(pool_output, [-1, pool_output.shape[-1]]))
        dense2 = Dense(units=pool_output.shape[-1], activation='relu')(dense1)
        weights = Reshape(pool_output.shape[:-1] + [pool_output.shape[-1]])(dense2)
        output_tensor = Multiply()([pool_output, weights])
        return output_tensor

    block2_output = block2(transition_conv)

    # Branch
    branch_output = Input(shape=(32, 32, 64))
    
    # Combine outputs
    combined_output = keras.layers.add([block2_output, branch_output])

    # Final Classification Layer
    output_layer = Dense(units=10, activation='softmax')(combined_output)

    model = keras.Model(inputs=[input_layer, branch_output], outputs=output_layer)

    return model