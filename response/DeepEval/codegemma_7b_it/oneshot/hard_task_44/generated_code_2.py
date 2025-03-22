import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dropout

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split input into three groups
        group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Feature extraction with different kernel sizes
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group3)

        # Dropout to reduce overfitting
        drop_out = Dropout(rate=0.2)(conv1)

        # Concatenate outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])

        return output_tensor

    # Block 2
    def block2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        branch2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        branch3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3)

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_tensor)
        branch4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate outputs
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])

        return output_tensor

    # Block 1 output
    block1_output = block1(input_tensor)

    # Block 2 output
    block2_output = block2(block1_output)

    # Feature fusion
    concat_output = Concatenate()([block1_output, block2_output])

    # Batch normalization and flattening
    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model