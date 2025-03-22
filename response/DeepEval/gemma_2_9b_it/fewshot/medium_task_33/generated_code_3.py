import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def split_channels(input_tensor):
        return Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

    split_tensor = split_channels(input_layer)
    
    # Channel 1
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])

    # Channel 2
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    conv2_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])

    # Channel 3
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    conv3_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Concatenate outputs
    merged_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(merged_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model