import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Lambda, Flatten, Dense, MaxPooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        
        # Apply Dropout
        dropped_out = Dropout(rate=0.5)(Concatenate()([conv1, conv2, conv3]))
        
        return dropped_out

    # Block 2
    def block_2(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(branch3)
        
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model