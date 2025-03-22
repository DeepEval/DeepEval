import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    def first_block(input_tensor):
        # Split input along the channel dimension into 3 groups
        split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[0])
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_channels[1])
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_channels[2])

        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = first_block(input_tensor=input_layer)

    # Second Block
    def second_block(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    block2_output = second_block(input_tensor=block1_output)

    # Global Average Pooling and Dense layer
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model