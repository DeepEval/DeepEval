import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Depthwise Separable Convolutions
    def depthwise_block(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        branch3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        
        return Concatenate()([branch1, branch2, branch3])

    block1_output = depthwise_block(input_layer)

    # Second Block: Multiple Branches
    def branch_block(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)

        return Concatenate()([branch1, branch2, branch3])

    block2_output = branch_block(block1_output)

    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model