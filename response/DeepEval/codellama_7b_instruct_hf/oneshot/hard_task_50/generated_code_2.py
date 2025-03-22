import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Max pooling layers with different scales
    max_pooling_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    max_pooling_4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the output of each pooling layer
    flatten_1 = Flatten()(max_pooling_1)
    flatten_2 = Flatten()(max_pooling_2)
    flatten_4 = Flatten()(max_pooling_4)

    # Concatenate the flattened outputs
    concatenated_outputs = Concatenate()([flatten_1, flatten_2, flatten_4])

    # Dropout to mitigate overfitting
    dropout = Dropout(0.2)(concatenated_outputs)

    # Fully connected layer
    dense = Dense(units=128, activation='relu')(dropout)

    # Reshape the output to a 4D tensor
    reshaped_output = Reshape(target_shape=(4, 4, 16))(dense)

    # Block 2
    # Split the input into 4 groups along the last dimension
    split_1 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshaped_output)
    split_2 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshaped_output)
    split_3 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshaped_output)
    split_4 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshaped_output)

    # Separable convolutional layers with varying kernel sizes
    conv_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_1)
    conv_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_2)
    conv_3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(split_3)
    conv_4 = Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding='same')(split_4)

    # Concatenate the outputs of the separable convolutional layers
    concatenated_outputs_block_2 = Concatenate()([conv_1, conv_2, conv_3, conv_4])

    # Flatten the output of the block 2
    flattened_output = Flatten()(concatenated_outputs_block_2)

    # Fully connected layer
    dense_output = Dense(units=10, activation='softmax')(flattened_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model