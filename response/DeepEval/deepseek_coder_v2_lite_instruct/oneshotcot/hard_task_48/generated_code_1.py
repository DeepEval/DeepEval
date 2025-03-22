import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(x):
        # Split the input into three groups
        split1 = Lambda(lambda tensor: tf.split(tensor, 3, axis=3))(x)
        # Apply different convolutional layers to each group
        conv1_1 = Conv2D(64, (1, 1), activation='relu')(split1[0])
        conv1_2 = Conv2D(64, (3, 3), activation='relu')(split1[1])
        conv1_3 = Conv2D(64, (5, 5), activation='relu')(split1[2])
        # Batch normalization
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        conv1_3 = BatchNormalization()(conv1_3)
        # Concatenate the outputs of the three groups
        output_tensor = Concatenate(axis=3)([conv1_1, conv1_2, conv1_3])
        return output_tensor

    block1_output = block1(input_layer)

    # Block 2
    def block2(x):
        # Path 1: 1x1 convolution
        path1 = Conv2D(64, (1, 1), activation='relu')(x)
        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = Conv2D(64, (1, 1), activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=1)(x))
        # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions
        path3_1 = Conv2D(64, (1, 1), activation='relu')(x)
        path3_2 = Conv2D(64, (1, 3), activation='relu')(path3_1)
        path3_3 = Conv2D(64, (3, 1), activation='relu')(path3_1)
        path3 = Concatenate(axis=3)([path3_2, path3_3])
        # Path 4: 1x1 convolution followed by 3x3 convolution
        # Then split into 1x3 and 3x1 convolutions
        path4_1 = Conv2D(64, (1, 1), activation='relu')(x)
        path4_2 = Conv2D(64, (3, 3), activation='relu')(path4_1)
        path4_3 = Conv2D(64, (1, 3), activation='relu')(path4_2)
        path4_4 = Conv2D(64, (3, 1), activation='relu')(path4_2)
        path4 = Concatenate(axis=3)([path4_3, path4_4])
        # Concatenate the outputs of the four paths
        output_tensor = Concatenate(axis=3)([path1, path2, path3, path4])
        return output_tensor

    block2_output = block2(block1_output)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model