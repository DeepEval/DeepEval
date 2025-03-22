import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute
from tensorflow.keras.initializers import glorot_uniform

def block_1(input_tensor):
    # Split the input into three groups
    x1, x2, x3 = tf.split(input_tensor, 3, axis=3)

    # Apply 1x1 convolutional layers to each group
    x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x1)
    x2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x2)
    x3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x3)

    # Concatenate the outputs of the three groups
    output = Concatenate()([x1, x2, x3])

    return output

def block_2(input_tensor):
    # Get the shape of the input tensor
    input_shape = keras.backend.int_shape(input_tensor)

    # Reshape the input tensor
    x = Reshape((input_shape[1], input_shape[2], input_shape[3] // 3, 3))(input_tensor)

    # Swap the third and fourth dimensions of the reshaped tensor
    x = Permute((1, 2, 4, 3))(x)

    # Reshape the tensor back to its original shape
    output = Reshape((input_shape[1], input_shape[2], input_shape[3]))(x)

    return output

def block_3(input_tensor):
    # Apply a 3x3 depthwise separable convolution
    output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                  activation='relu', use_bias=False)(input_tensor)

    return output

def block_4(input_tensor):
    # Apply a 1x1 convolutional layer
    output = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same',
                  activation='relu', use_bias=False)(input_tensor)

    return output

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = block_1(input_layer)
    x = block_2(x)
    x = block_3(x)
    x = block_4(x)

    # Branch path
    y = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    y = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    y = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(y)

    # Concatenate the outputs of the main path and the branch path
    output = Concatenate()([x, y])

    # Fully connected layer
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output)

    return model