import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Block 1
    # Split the input into two groups
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(conv1)

    # Operations on the first group
    conv1_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    conv1_depthwise = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(conv1_1x1)
    conv1_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_depthwise)

    # Operations on the second group
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])

    # Merge the outputs
    merged = Concatenate()([conv1_1x1_2, conv2])

    # Block 2
    # Reshape the input for channel shuffling
    input_shape = keras.backend.int_shape(merged)
    reshaped = Lambda(lambda x: tf.reshape(x, shape=[-1, input_shape[1], input_shape[2], input_shape[3] // 2, 2]))(merged)

    # Swap the third and fourth dimensions
    swapped = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2, 4]))(reshaped)

    # Reshape back to original shape for channel shuffling
    shuffled = Lambda(lambda x: tf.reshape(x, shape=[-1, input_shape[1], input_shape[2], input_shape[3]]))(swapped)

    # Flatten and fully connected layer
    flatten = Flatten()(shuffled)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model