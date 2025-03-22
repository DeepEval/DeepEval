import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Lambda, Concatenate, AveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Depthwise Separable Convolutions
    def first_block(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        batchnorm = BatchNormalization()(conv1)
        batchnorm = BatchNormalization()(conv2)
        batchnorm = BatchNormalization()(conv3)
        output_tensor = Concatenate()([batchnorm, batchnorm, batchnorm])
        return output_tensor

    block1_output = first_block(input_tensor=input_layer)

    # Second Block: Branching Feature Extraction
    def second_block(input_tensor):
        branch1 =  DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

        branch2 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = DepthwiseConv2D(kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = DepthwiseConv2D(kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)

        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    block2_output = second_block(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model