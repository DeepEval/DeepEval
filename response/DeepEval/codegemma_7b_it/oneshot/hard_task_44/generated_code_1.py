import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])
        concat = Concatenate()([conv1, conv2, conv3])
        dropout = Dropout(0.2)(concat)

        return dropout

    block1_output = block1(input_layer)

    # Block 2
    def block2(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        concat = Concatenate()([branch1, branch2, branch3, branch4])

        return concat

    block2_output = block2(block1_output)

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model