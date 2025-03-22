import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow import tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def split_and_process(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        return Concatenate()([conv1, conv2, conv3])

    block1_output = split_and_process(input_layer)

    # Second Block
    def second_block(input_tensor):
        branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        return Concatenate()([branch1, branch2, branch3])

    block2_output = second_block(block1_output)

    # Global Average Pooling and Classification
    gap = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(gap)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model