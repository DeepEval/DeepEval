import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape
from keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # Input shape for CIFAR-10
    input_layer = Input(shape=input_shape)

    # Block 1: Splitting and Convolution
    def block1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[0])
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[2])
        fuse1 = Concatenate(axis=-1)([conv1, conv2, conv3])
        return fuse1

    # Block 2: Channel Shuffling
    def block2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        reshape_tensor = Reshape((shape[1] // 2, shape[2] * 2, 2))(input_tensor)
        swap = keras.backend.permute_dimensions(reshape_tensor, (0, 2, 1, 3))
        reshape_output = Reshape((shape[1] // 2, shape[2], 2))(swap)
        return reshape_output

    # Block 3: Depthwise Separable Convolution
    def block3(input_tensor):
        dwconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return dwconv

    # Main Path
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block3_output = block3(block2_output)

    # Branch Path
    average_pool = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    branch_output = block1(average_pool)

    # Concatenation
    concatenated = Concatenate(axis=-1)([block1_output, branch_output, block3_output])

    # Fully Connected Layer
    dense = Dense(units=128, activation='relu')(concatenated)
    output = Dense(units=10, activation='softmax')(dense)

    # Model Construction
    model = Model(inputs=input_layer, outputs=output)

    return model

# Instantiate the model
model = dl_model()
model.summary()