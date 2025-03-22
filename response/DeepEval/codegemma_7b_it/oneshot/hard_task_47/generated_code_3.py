import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def conv_block(input_tensor, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        x = keras.activations.relu(x)
        return x

    split_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    conv1 = conv_block(split_input[0], filters=32, kernel_size=(1, 1))
    conv2 = conv_block(split_input[1], filters=32, kernel_size=(3, 3))
    conv3 = conv_block(split_input[2], filters=32, kernel_size=(5, 5))
    concat1 = Concatenate()([conv1, conv2, conv3])

    # Second block
    def branch_conv(input_tensor, kernel_sizes):
        x = input_tensor
        for kernel_size in kernel_sizes:
            x = Conv2D(filters=64, kernel_size=kernel_size, padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = keras.activations.relu(x)
        return x

    branch1 = branch_conv(concat1, kernel_sizes=[(1, 1)])
    branch2 = branch_conv(concat1, kernel_sizes=[(3, 3)])
    branch3 = branch_conv(concat1, kernel_sizes=[(1, 7), (7, 1)])
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(concat1)
    concat2 = Concatenate()([branch1, branch2, branch3, branch4])

    # Fully connected layers
    flatten_layer = Flatten()(concat2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model