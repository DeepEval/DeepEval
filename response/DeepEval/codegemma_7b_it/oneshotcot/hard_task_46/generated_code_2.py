import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
    concat_block1 = Concatenate()([conv1, conv2, conv3])

    # Second block
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_block1)

    branch1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv4)

    concat_block2 = Concatenate()([branch1, branch2, branch3])

    # Global average pooling and fully connected layer
    gap = GlobalAveragePooling2D()(concat_block2)
    output_layer = Dense(units=10, activation='softmax')(gap)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model