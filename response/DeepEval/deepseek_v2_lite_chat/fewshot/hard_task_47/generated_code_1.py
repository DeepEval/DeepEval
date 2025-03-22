import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, BatchNormalization, Add, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split input into three groups along the last dimension
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Depthwise separable convolutional layers with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

        # Batch normalization
        bn1 = BatchNormalization()(conv1)
        bn2 = BatchNormalization()(conv2)
        bn3 = BatchNormalization()(conv3)

        # Concatenate outputs
        concat_tensor = Concatenate()(outputs=[bn1, bn2, bn3])

        return concat_tensor

    def block_2(input_tensor):
        # Three branches for feature extraction
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Average pooling branch
        avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True))(input_tensor)

        # Concatenate all branches
        concat_tensor = Concatenate()(outputs=[branch1, branch2, branch3, avg_pool])

        # Fully connected layers
        flatten = Flatten()(concat_tensor)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        return output_layer

    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)

    return model