import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer and reshape it to (height, width, groups, channels_per_group)
    input_layer = Input(shape=(32, 32, 3))
    reshaped_input = tf.reshape(input_layer, (32, 32, 3, 1))

    # Swap the third and fourth dimensions using a permutation operation to enable channel shuffling
    shuffled_input = tf.transpose(reshaped_input, (0, 1, 3, 2))

    # Reshape the input back to its original shape
    reshaped_shuffled_input = tf.reshape(shuffled_input, (32, 32, 1, 3))

    # Define the convolutional layer and pooling layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(reshaped_shuffled_input)
    pooling_layer = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_layer)

    # Define the batch normalization layer
    batch_normalization_layer = BatchNormalization()(pooling_layer)

    # Define the flatten layer
    flatten_layer = Flatten()(batch_normalization_layer)

    # Define the fully connected layers
    dense_layer_1 = Dense(units=128, activation='relu')(flatten_layer)
    dense_layer_2 = Dense(units=64, activation='relu')(dense_layer_1)
    output_layer = Dense(units=10, activation='softmax')(dense_layer_2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model