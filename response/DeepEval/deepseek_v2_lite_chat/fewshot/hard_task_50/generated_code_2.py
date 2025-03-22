import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        flatten2 = Flatten()(maxpool2)
        flatten3 = Flatten()(maxpool3)
        dropout = Dropout(rate=0.5)(flatten1)
        merged = Concatenate()([dropout, flatten2, flatten3])
        return merged

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        conv1 = Dense(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Dense(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Dense(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        conv4 = Dense(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(inputs_groups[3])
        concatenated = Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
        return concatenated

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    flattened = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model