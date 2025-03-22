import keras
from keras.layers import Input, Lambda, Split, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: split input into three groups along last dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    concatenated = Concatenate()([group1, group2, group3])

    # Second block: multiple branches for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concatenated)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concatenated)
    branch4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    concatenated_branches = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten and fully connected layers
    flattened = Flatten()(concatenated_branches)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model